"""
Extraction Shooter Matchmaking Evaluation Framework
====================================================
Complete codebase for evaluating matchmaking strategies.

Components:
- Raid simulation (from previous work)
- Persistent players with running stats
- Multiple matchmaking strategies (baselines)
- Multi-objective reward function (Pareto)
- Episode runner and evaluation framework
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Callable
import matplotlib.pyplot as plt
import json
from abc import ABC, abstractmethod


# =============================================================================
# PLAYER CLASSIFICATION
# =============================================================================

def classify_player(aggression: float) -> str:
    """Classify player by aggression score."""
    if aggression < 0.4:
        return "passive"
    elif aggression > 0.6:
        return "aggressive"
    else:
        return "neutral"


# =============================================================================
# PERSISTENT PLAYER
# =============================================================================

@dataclass
class PersistentPlayer:
    """Player with running statistics."""
    id: int
    aggression: float = 0.5
    
    total_raids: int = 0
    total_extractions: int = 0
    total_deaths: int = 0
    total_kills: int = 0
    total_stash: float = 0.0
    total_damage_dealt: float = 0.0
    total_damage_received: float = 0.0
    
    # Running average aggression (for classification)
    aggression_sum: float = 0.0
    aggression_count: int = 0
    
    @property
    def running_aggression(self) -> float:
        """Running average of aggression used in raids."""
        if self.aggression_count == 0:
            return self.aggression
        return round(self.aggression_sum / self.aggression_count, 3)
    
    @property
    def classification(self) -> str:
        return classify_player(self.running_aggression)
    
    @property
    def extraction_rate(self) -> float:
        if self.total_raids == 0:
            return 0.0
        return round(self.total_extractions / self.total_raids, 3)
    
    @property
    def avg_stash(self) -> float:
        if self.total_extractions == 0:
            return 0.0
        return round(self.total_stash / self.total_extractions, 1)
    
    @property
    def kd_ratio(self) -> float:
        if self.total_deaths == 0:
            return float(self.total_kills)
        return round(self.total_kills / self.total_deaths, 3)
    
    @property
    def damage_ratio(self) -> float:
        if self.total_damage_received == 0:
            return 1.0
        return round(self.total_damage_dealt / self.total_damage_received, 3)
    
    def get_state_vector(self) -> np.ndarray:
        """Markov state for RL."""
        return np.array([
            self.running_aggression,
            self.extraction_rate,
            min(self.avg_stash / 200000, 1.0),
            min(self.kd_ratio / 5, 1.0),
            min(self.damage_ratio / 3, 1.0),
            min(self.total_raids / 100, 1.0),
        ], dtype=np.float32)
    
    def get_raid_aggression(self, noise_std: float = 0.05) -> float:
        """Get aggression for raid with noise."""
        noisy = self.aggression + np.random.normal(0, noise_std)
        return round(np.clip(noisy, 0.0, 1.0), 3)
    
    def record_raid(self, extracted: bool, stash: float, damage_dealt: float,
                    damage_received: float, kills: int, aggression_used: float):
        """Update stats after raid."""
        self.total_raids += 1
        self.total_damage_dealt += damage_dealt
        self.total_damage_received += damage_received
        self.total_kills += kills
        self.aggression_sum += aggression_used
        self.aggression_count += 1
        
        if extracted:
            self.total_extractions += 1
            self.total_stash += stash
        else:
            self.total_deaths += 1
    
    def update_aggression(self, extracted: bool, kills: int, damage_dealt: float,
                          damage_received: float, aggression_used: float, 
                          learning_rate: float = 0.03):
        """
        Evolve base aggression with balanced rules.
        
        Key design: Both passive and aggressive playstyles should be stable
        if successful. Dying should push you toward your natural playstyle
        (more cautious if passive, reconsider if aggressive).
        """
        delta = 0.0
        
        # Outcome-based adjustment
        if extracted:
            if kills > 0:
                # Aggressive extraction - reinforce aggression
                delta += learning_rate * (0.4 + kills * 0.15)
            else:
                # Peaceful extraction - reinforce passivity
                delta -= learning_rate * 0.6
        else:
            # Died - learn from failure
            if aggression_used > 0.5:
                # Was aggressive but died → failed hunt, become cautious
                delta -= learning_rate * 0.5
            else:
                # Was passive but died → should be MORE passive/careful
                delta -= learning_rate * 0.3
        
        # Damage-based adjustment (behavioral reinforcement)
        # High damage dealer → drift aggressive
        # Low damage dealer → drift passive
        if damage_dealt > 100:
            delta += learning_rate * 0.2
        elif damage_dealt < 30:
            delta -= learning_rate * 0.2
        
        # Apply with mean reversion toward 0.5 for neutrals
        # This prevents runaway extremes
        if 0.4 < self.aggression < 0.6:
            # Neutral players have slight pull toward center
            delta += learning_rate * 0.1 * (0.5 - self.aggression)
        
        self.aggression = round(np.clip(self.aggression + delta, 0.0, 1.0), 3)
    
    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'aggression': self.aggression,
            'total_raids': self.total_raids,
            'total_extractions': self.total_extractions,
            'total_deaths': self.total_deaths,
            'total_kills': self.total_kills,
            'total_stash': round(self.total_stash, 1),
            'total_damage_dealt': round(self.total_damage_dealt, 1),
            'total_damage_received': round(self.total_damage_received, 1),
            'aggression_sum': round(self.aggression_sum, 3),
            'aggression_count': self.aggression_count
        }
    
    @staticmethod
    def from_dict(d: dict) -> 'PersistentPlayer':
        p = PersistentPlayer(id=d['id'], aggression=d['aggression'])
        p.total_raids = d['total_raids']
        p.total_extractions = d['total_extractions']
        p.total_deaths = d['total_deaths']
        p.total_kills = d['total_kills']
        p.total_stash = d['total_stash']
        p.total_damage_dealt = d['total_damage_dealt']
        p.total_damage_received = d['total_damage_received']
        p.aggression_sum = d.get('aggression_sum', 0.0)
        p.aggression_count = d.get('aggression_count', 0)
        return p
    
    def copy(self) -> 'PersistentPlayer':
        """Create a deep copy."""
        return PersistentPlayer.from_dict(self.to_dict())


# =============================================================================
# PLAYER POOL
# =============================================================================

class PlayerPool:
    """Pool of persistent players."""
    
    def __init__(self, num_players: int = 120, default_aggression: float = None,
                 diverse: bool = False):
        """
        Create player pool.
        
        Args:
            num_players: Number of players
            default_aggression: If set, all players start with this value
            diverse: If True, initialize with uniform distribution [0, 1]
        """
        self.players: Dict[int, PersistentPlayer] = {}
        
        for i in range(num_players):
            if diverse:
                # Uniform distribution across full range
                aggr = round(np.random.uniform(0.05, 0.95), 3)
            elif default_aggression is not None:
                aggr = default_aggression
            else:
                aggr = 0.5
            
            self.players[i] = PersistentPlayer(id=i, aggression=aggr)
    
    def get_player(self, pid: int) -> PersistentPlayer:
        return self.players[pid]
    
    def get_all_players(self) -> List[PersistentPlayer]:
        return list(self.players.values())
    
    def get_players_by_classification(self) -> Dict[str, List[PersistentPlayer]]:
        """Group players by classification."""
        groups = {'passive': [], 'neutral': [], 'aggressive': []}
        for p in self.players.values():
            groups[p.classification].append(p)
        return groups
    
    def sample_random(self, n: int) -> List[PersistentPlayer]:
        ids = np.random.choice(list(self.players.keys()), size=n, replace=False)
        return [self.players[i] for i in ids]
    
    def get_stats(self) -> dict:
        players = self.get_all_players()
        aggr = [p.running_aggression for p in players]
        groups = self.get_players_by_classification()
        
        return {
            'total_players': len(players),
            'passive_count': len(groups['passive']),
            'neutral_count': len(groups['neutral']),
            'aggressive_count': len(groups['aggressive']),
            'aggression_mean': round(np.mean(aggr), 3),
            'aggression_std': round(np.std(aggr), 3),
            'aggression_min': round(np.min(aggr), 3),
            'aggression_max': round(np.max(aggr), 3),
        }
    
    def save(self, filepath: str):
        data = {'players': {pid: p.to_dict() for pid, p in self.players.items()}}
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @staticmethod
    def load(filepath: str) -> 'PlayerPool':
        with open(filepath, 'r') as f:
            data = json.load(f)
        pool = PlayerPool(num_players=0)
        pool.players = {int(pid): PersistentPlayer.from_dict(p) for pid, p in data['players'].items()}
        return pool
    
    def copy(self) -> 'PlayerPool':
        """Deep copy the pool."""
        new_pool = PlayerPool(num_players=0)
        new_pool.players = {pid: p.copy() for pid, p in self.players.items()}
        return new_pool


# =============================================================================
# RAID SIMULATION (Compact)
# =============================================================================

@dataclass
class LootZone:
    x: float
    y: float
    radius: float
    total_items: int
    remaining_items: int
    value_multiplier: float
    
    def loot_item(self) -> Optional[float]:
        if self.remaining_items <= 0:
            return None
        self.remaining_items -= 1
        return np.random.exponential(400 * self.value_multiplier)
    
    def is_empty(self) -> bool:
        return self.remaining_items <= 0
    
    def contains_point(self, x: float, y: float) -> bool:
        return (x - self.x)**2 + (y - self.y)**2 <= self.radius**2
    
    def reset(self):
        self.remaining_items = self.total_items


@dataclass
class ExtractionPoint:
    x: float
    y: float
    radius: float = 5.0
    name: str = ""
    cooldown: int = 0
    
    def contains_point(self, x: float, y: float) -> bool:
        return (x - self.x)**2 + (y - self.y)**2 <= self.radius**2
    
    def is_available(self) -> bool:
        return self.cooldown <= 0
    
    def start_cooldown(self, ticks: int):
        self.cooldown = ticks
    
    def tick_cooldown(self):
        if self.cooldown > 0:
            self.cooldown -= 1
    
    def reset(self):
        self.cooldown = 0


@dataclass
class RaidPlayer:
    """Transient player for single raid."""
    id: int
    persistent_id: int
    x: float
    y: float
    aggression: float
    
    hp: float = 100.0
    stash: float = 0.0
    alive: bool = True
    extracted: bool = False
    speed: float = 2.0
    
    target_x: float = 0.0
    target_y: float = 0.0
    target_type: str = "loot"
    
    extraction_ticks: int = 0
    extracting_at: Optional[str] = None
    
    combat_cooldown: int = 0
    no_fight_cooldowns: dict = field(default_factory=dict)
    in_combat_with: Optional[int] = None
    is_attacker: bool = False
    combat_ticks: int = 0
    
    damage_dealt: float = 0.0
    damage_received: float = 0.0
    kills: int = 0
    
    def is_alive(self) -> bool:
        return self.alive and self.hp > 0
    
    def is_extracting(self) -> bool:
        return self.extracting_at is not None
    
    def is_in_combat(self) -> bool:
        return self.in_combat_with is not None
    
    def can_fight(self, other_id: int) -> bool:
        if self.combat_cooldown > 0 or self.in_combat_with is not None:
            return False
        return other_id not in self.no_fight_cooldowns or self.no_fight_cooldowns[other_id] <= 0
    
    def decide_to_fight(self) -> bool:
        return np.random.random() < self.aggression
    
    def deal_damage(self) -> float:
        return np.random.uniform(1 + 4 * self.aggression, 5 + 15 * self.aggression)
    
    def take_damage(self, dmg: float):
        self.hp -= dmg
        self.damage_received += dmg
        if self.hp <= 0:
            self.hp = 0
            self.alive = False
    
    def heal(self, amount: float):
        self.hp = min(100.0, self.hp + amount)
    
    def distance_to(self, x: float, y: float) -> float:
        return np.sqrt((self.x - x)**2 + (self.y - y)**2)
    
    def move_toward(self, tx: float, ty: float, map_radius: float):
        dx, dy = tx - self.x, ty - self.y
        dist = np.sqrt(dx**2 + dy**2)
        if dist < self.speed:
            self.x, self.y = tx, ty
        else:
            self.x += self.speed * dx / dist
            self.y += self.speed * dy / dist
        dist_center = np.sqrt(self.x**2 + self.y**2)
        if dist_center > map_radius:
            scale = (map_radius - 0.1) / dist_center
            self.x *= scale
            self.y *= scale
    
    def update_cooldowns(self):
        if self.combat_cooldown > 0:
            self.combat_cooldown -= 1
        expired = [pid for pid, cd in self.no_fight_cooldowns.items() if cd <= 1]
        for pid in expired:
            del self.no_fight_cooldowns[pid]
        for pid in self.no_fight_cooldowns:
            self.no_fight_cooldowns[pid] -= 1
    
    def start_combat(self, other_id: int, as_attacker: bool):
        self.in_combat_with = other_id
        self.is_attacker = as_attacker
        self.combat_ticks = 0
    
    def end_combat(self):
        self.in_combat_with = None
        self.is_attacker = False
        self.combat_ticks = 0
    
    def set_no_fight_cooldown(self, other_id: int, steps: int = 30):
        self.no_fight_cooldowns[other_id] = steps


@dataclass
class MapConfig:
    radius: float = 100.0
    num_loot_zones: int = 8
    loot_zone_min_radius: float = 5.0
    loot_zone_max_radius: float = 15.0
    items_per_unit_area: float = 0.3
    extraction_radius: float = 5.0
    spawn_radius: float = 90.0
    player_sight_radius: float = 15.0


@dataclass
class RaidConfig:
    max_ticks: int = 500
    sight_radius: float = 15.0
    no_fight_cooldown: int = 30
    extraction_buffer: float = 1.5
    extraction_time: int = 30
    extraction_cooldown: int = 30
    combat_max_ticks: int = 15
    flee_hp_threshold: float = 35.0
    flee_success_base: float = 0.3
    post_combat_cooldown: int = 15
    heal_on_kill: float = 5.0


class GameMap:
    def __init__(self, config: MapConfig = None):
        self.config = config or MapConfig()
        self.loot_zones: List[LootZone] = []
        self.extraction_points: List[ExtractionPoint] = []
        self._generate_map()
    
    def _generate_map(self):
        self.loot_zones = []
        max_dist = 0.5 * self.config.radius
        
        for _ in range(self.config.num_loot_zones):
            for _ in range(100):
                angle = np.random.uniform(0, 2 * np.pi)
                dist = np.random.uniform(0, max_dist)
                x, y = dist * np.cos(angle), dist * np.sin(angle)
                radius = np.random.uniform(self.config.loot_zone_min_radius, self.config.loot_zone_max_radius)
                
                if all(np.sqrt((x-z.x)**2 + (y-z.y)**2) >= radius + z.radius + 5 for z in self.loot_zones):
                    if dist + radius <= max_dist + 10:
                        area = np.pi * radius**2
                        items = max(3, int(area * self.config.items_per_unit_area))
                        self.loot_zones.append(LootZone(x, y, radius, items, items, radius/self.config.loot_zone_min_radius))
                        break
        
        d = 0.9 * self.config.radius
        self.extraction_points = [
            ExtractionPoint(0, d, self.config.extraction_radius, "N"),
            ExtractionPoint(0, -d, self.config.extraction_radius, "S"),
            ExtractionPoint(d, 0, self.config.extraction_radius, "E"),
            ExtractionPoint(-d, 0, self.config.extraction_radius, "W"),
        ]
    
    def get_spawn_positions(self, n: int) -> List[Tuple[float, float]]:
        angles = np.linspace(0, 2*np.pi, n, endpoint=False) + np.pi/n
        return [(self.config.spawn_radius * np.cos(a), self.config.spawn_radius * np.sin(a)) for a in angles]
    
    def get_closest_loot_zone(self, x: float, y: float) -> Optional[LootZone]:
        valid = [z for z in self.loot_zones if not z.is_empty()]
        if not valid:
            return None
        return min(valid, key=lambda z: np.sqrt((z.x-x)**2 + (z.y-y)**2))
    
    def get_closest_extraction(self, x: float, y: float, prefer_available: bool = True) -> ExtractionPoint:
        if prefer_available:
            available = [e for e in self.extraction_points if e.is_available()]
            if available:
                return min(available, key=lambda e: np.sqrt((e.x-x)**2 + (e.y-y)**2))
        return min(self.extraction_points, key=lambda e: np.sqrt((e.x-x)**2 + (e.y-y)**2))
    
    def tick_extractions(self):
        for ext in self.extraction_points:
            ext.tick_cooldown()
    
    def reset(self):
        for z in self.loot_zones:
            z.reset()
        for e in self.extraction_points:
            e.reset()


class Raid:
    """Single raid simulation."""
    
    def __init__(self, game_map: GameMap, players: List[RaidPlayer], config: RaidConfig = None):
        self.map = game_map
        self.players = players
        self.config = config or RaidConfig()
        self.tick = 0
    
    def get_alive(self) -> List[RaidPlayer]:
        return [p for p in self.players if p.is_alive() and not p.extracted]
    
    def get_visible(self, player: RaidPlayer) -> List[RaidPlayer]:
        return [p for p in self.get_alive() if p.id != player.id and player.distance_to(p.x, p.y) <= self.config.sight_radius]
    
    def should_extract(self, p: RaidPlayer) -> bool:
        ext = self.map.get_closest_extraction(p.x, p.y, True)
        ticks_needed = (p.distance_to(ext.x, ext.y) / p.speed + self.config.extraction_time) * self.config.extraction_buffer
        caution = 1 + (1 - p.aggression) * 0.5
        return (self.config.max_ticks - self.tick) <= ticks_needed * caution
    
    def select_target(self, p: RaidPlayer) -> Tuple[float, float, str]:
        if self.should_extract(p):
            ext = self.map.get_closest_extraction(p.x, p.y, True)
            return ext.x, ext.y, "extraction"
        
        if p.aggression > 0.7:
            visible = self.get_visible(p)
            if visible:
                t = max(visible, key=lambda x: x.stash)
                return t.x, t.y, "player"
            zones = [z for z in self.map.loot_zones if not z.is_empty()]
            if zones:
                z = max(zones, key=lambda x: x.radius)
                return z.x, z.y, "loot"
        
        zone = self.map.get_closest_loot_zone(p.x, p.y)
        if zone:
            return zone.x, zone.y, "loot"
        
        ext = self.map.get_closest_extraction(p.x, p.y, True)
        return ext.x, ext.y, "extraction"
    
    def resolve_encounter(self, p1: RaidPlayer, p2: RaidPlayer):
        if not p1.can_fight(p2.id) or not p2.can_fight(p1.id):
            return
        
        f1, f2 = p1.decide_to_fight(), p2.decide_to_fight()
        
        if not f1 and not f2:
            p1.set_no_fight_cooldown(p2.id, self.config.no_fight_cooldown)
            p2.set_no_fight_cooldown(p1.id, self.config.no_fight_cooldown)
            return
        
        if f1 and not f2:
            a, d = p1, p2
        elif f2 and not f1:
            a, d = p2, p1
        else:
            a, d = (p1, p2) if np.random.random() < p1.aggression/(p1.aggression+p2.aggression+0.01) else (p2, p1)
        
        a.start_combat(d.id, True)
        d.start_combat(a.id, False)
    
    def process_combat(self, p: RaidPlayer):
        if not p.is_in_combat():
            return
        
        other = next((x for x in self.players if x.id == p.in_combat_with), None)
        if not other or not other.is_alive():
            p.end_combat()
            return
        
        if not p.is_attacker:
            return
        
        p.combat_ticks += 1
        other.combat_ticks += 1
        
        dmg = p.deal_damage()
        other.take_damage(dmg)
        p.damage_dealt += dmg
        
        if not other.is_alive():
            p.stash += other.stash
            other.stash = 0
            p.kills += 1
            p.heal(self.config.heal_on_kill)
            p.end_combat()
            other.end_combat()
            p.combat_cooldown = self.config.post_combat_cooldown
            return
        
        if other.hp < self.config.flee_hp_threshold:
            if np.random.random() < self.config.flee_success_base + (1-other.aggression)*0.4:
                p.end_combat()
                other.end_combat()
                p.combat_cooldown = self.config.post_combat_cooldown
                other.combat_cooldown = self.config.post_combat_cooldown
                p.set_no_fight_cooldown(other.id, 50)
                other.set_no_fight_cooldown(p.id, 50)
                return
        
        if p.combat_ticks >= self.config.combat_max_ticks:
            p.end_combat()
            other.end_combat()
            p.combat_cooldown = self.config.post_combat_cooldown
            other.combat_cooldown = self.config.post_combat_cooldown
            p.set_no_fight_cooldown(other.id, 40)
            other.set_no_fight_cooldown(p.id, 40)
            return
        
        p.is_attacker = False
        other.is_attacker = True
    
    def tick_player(self, p: RaidPlayer):
        if not p.is_alive() or p.extracted:
            return
        
        p.update_cooldowns()
        
        if p.is_in_combat():
            if p.is_extracting():
                p.extraction_ticks = 0
                p.extracting_at = None
            self.process_combat(p)
            return
        
        for other in self.get_visible(p):
            if other.is_alive() and not other.is_in_combat() and p.can_fight(other.id):
                self.resolve_encounter(p, other)
                if not p.is_alive() or p.is_in_combat():
                    return
        
        ext = next((e for e in self.map.extraction_points if e.contains_point(p.x, p.y)), None)
        if ext and p.target_type == "extraction":
            if ext.is_available():
                if p.extracting_at == ext.name:
                    p.extraction_ticks += 1
                else:
                    p.extracting_at = ext.name
                    p.extraction_ticks = 1
                if p.extraction_ticks >= self.config.extraction_time:
                    p.extracted = True
                    ext.start_cooldown(self.config.extraction_cooldown)
                return
            else:
                p.extracting_at = None
                p.extraction_ticks = 0
        else:
            if p.is_extracting():
                p.extracting_at = None
                p.extraction_ticks = 0
        
        p.target_x, p.target_y, p.target_type = self.select_target(p)
        
        zone = next((z for z in self.map.loot_zones if z.contains_point(p.x, p.y)), None)
        if zone and not zone.is_empty() and p.target_type == "loot":
            v = zone.loot_item()
            if v:
                p.stash += v
            return
        
        p.move_toward(p.target_x, p.target_y, self.map.config.radius)
    
    def run_tick(self):
        self.map.tick_extractions()
        alive = self.get_alive()
        np.random.shuffle(alive)
        for p in alive:
            self.tick_player(p)
        self.tick += 1
    
    def run(self) -> List[dict]:
        """Run raid, return per-player results."""
        while self.tick < self.config.max_ticks and self.get_alive():
            self.run_tick()
        
        return [{
            'persistent_id': p.persistent_id,
            'extracted': p.extracted,
            'stash': round(p.stash, 1) if p.extracted else 0,
            'damage_dealt': round(p.damage_dealt, 1),
            'damage_received': round(p.damage_received, 1),
            'kills': p.kills,
            'aggression_used': round(p.aggression, 3)
        } for p in self.players]


# =============================================================================
# RAID RUNNER
# =============================================================================

class RaidRunner:
    """Runs raids with persistent players."""
    
    def __init__(self, map_config: MapConfig = None, raid_config: RaidConfig = None):
        self.map_config = map_config or MapConfig()
        self.raid_config = raid_config or RaidConfig()
        self.game_map = GameMap(self.map_config)
    
    def run_single_raid(self, players: List[PersistentPlayer], 
                        noise_std: float = 0.05) -> List[dict]:
        """Run one raid, return results."""
        self.game_map.reset()
        spawns = self.game_map.get_spawn_positions(len(players))
        
        raid_players = [
            RaidPlayer(id=i, persistent_id=pp.id, x=x, y=y, 
                      aggression=pp.get_raid_aggression(noise_std))
            for i, (pp, (x, y)) in enumerate(zip(players, spawns))
        ]
        
        return Raid(self.game_map, raid_players, self.raid_config).run()
    
    def run_averaged_raid(self, players: List[PersistentPlayer],
                          num_repeats: int = 10, noise_std: float = 0.05) -> Dict[int, dict]:
        """Run raid multiple times, average results."""
        acc = {pp.id: {'extracted': 0, 'stash': 0, 'dmg_dealt': 0, 
                       'dmg_recv': 0, 'kills': 0, 'aggr': 0}
               for pp in players}
        
        for _ in range(num_repeats):
            for r in self.run_single_raid(players, noise_std):
                pid = r['persistent_id']
                acc[pid]['extracted'] += 1 if r['extracted'] else 0
                acc[pid]['stash'] += r['stash']
                acc[pid]['dmg_dealt'] += r['damage_dealt']
                acc[pid]['dmg_recv'] += r['damage_received']
                acc[pid]['kills'] += r['kills']
                acc[pid]['aggr'] += r['aggression_used']
        
        return {
            pid: {
                'extracted': a['extracted'] > num_repeats / 2,
                'stash': round(a['stash'] / num_repeats, 1),
                'damage_dealt': round(a['dmg_dealt'] / num_repeats, 1),
                'damage_received': round(a['dmg_recv'] / num_repeats, 1),
                'kills': round(a['kills'] / num_repeats, 2),
                'aggression_used': round(a['aggr'] / num_repeats, 3)
            }
            for pid, a in acc.items()
        }


# =============================================================================
# REWARD FUNCTION (Multi-Pareto)
# =============================================================================

def compute_lobby_reward(raid_results: List[dict], 
                         player_classifications: Dict[int, str]) -> dict:
    """
    Compute multi-objective reward for a lobby.
    
    Returns detailed metrics and combined Pareto reward.
    Both passive AND aggressive players must succeed for high reward.
    """
    passive_results = []
    aggressive_results = []
    neutral_results = []
    
    for r in raid_results:
        pid = r['persistent_id']
        cls = player_classifications.get(pid, 'neutral')
        if cls == 'passive':
            passive_results.append(r)
        elif cls == 'aggressive':
            aggressive_results.append(r)
        else:
            neutral_results.append(r)
    
    # Passive score: extraction rate × normalized stash
    if passive_results:
        passive_extracted = sum(1 for r in passive_results if r['extracted'])
        passive_extraction_rate = passive_extracted / len(passive_results)
        passive_stash = sum(r['stash'] for r in passive_results if r['extracted'])
        passive_avg_stash = passive_stash / max(1, passive_extracted)
        passive_stash_score = min(passive_avg_stash / 100000, 1.0)
        passive_score = passive_extraction_rate * (0.5 + 0.5 * passive_stash_score)
    else:
        passive_extraction_rate = 0.0
        passive_avg_stash = 0.0
        passive_score = 0.0
    
    # Aggressive score: kills per player × extraction rate
    if aggressive_results:
        aggressive_kills = sum(r['kills'] for r in aggressive_results)
        aggressive_kills_per_player = aggressive_kills / len(aggressive_results)
        aggressive_extracted = sum(1 for r in aggressive_results if r['extracted'])
        aggressive_extraction_rate = aggressive_extracted / len(aggressive_results)
        aggressive_kill_score = min(aggressive_kills_per_player / 2, 1.0)
        aggressive_score = aggressive_kill_score * (0.5 + 0.5 * aggressive_extraction_rate)
    else:
        aggressive_kills = 0
        aggressive_kills_per_player = 0.0
        aggressive_extraction_rate = 0.0
        aggressive_score = 0.0
    
    # Combined Pareto reward (geometric mean)
    if passive_score > 0 and aggressive_score > 0:
        pareto_reward = np.sqrt(passive_score * aggressive_score)
    else:
        pareto_reward = 0.0
    
    return {
        'pareto_reward': round(pareto_reward, 3),
        'passive_score': round(passive_score, 3),
        'aggressive_score': round(aggressive_score, 3),
        'passive_count': len(passive_results),
        'aggressive_count': len(aggressive_results),
        'neutral_count': len(neutral_results),
        'passive_extraction_rate': round(passive_extraction_rate, 3),
        'passive_avg_stash': round(passive_avg_stash, 1),
        'aggressive_kills_per_player': round(aggressive_kills_per_player, 2),
        'aggressive_extraction_rate': round(aggressive_extraction_rate, 3),
    }


# =============================================================================
# MATCHMAKING STRATEGIES (Baselines)
# =============================================================================

class Matchmaker(ABC):
    """Abstract base class for matchmaking strategies."""
    
    @abstractmethod
    def create_lobbies(self, pool: PlayerPool, lobby_size: int, num_lobbies: int) -> List[List[PersistentPlayer]]:
        """Create lobbies from player pool."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass


class RandomMatchmaker(Matchmaker):
    """Completely random matchmaking."""
    
    @property
    def name(self) -> str:
        return "Random"
    
    def create_lobbies(self, pool: PlayerPool, lobby_size: int, num_lobbies: int) -> List[List[PersistentPlayer]]:
        all_players = pool.get_all_players()
        np.random.shuffle(all_players)
        
        lobbies = []
        for i in range(num_lobbies):
            start = i * lobby_size
            end = start + lobby_size
            if end <= len(all_players):
                lobbies.append(all_players[start:end])
        
        return lobbies


class PolarizedMatchmaker(Matchmaker):
    """
    Segregated matchmaking - what industry does.
    Passive players with passive, aggressive with aggressive.
    """
    
    @property
    def name(self) -> str:
        return "Polarized"
    
    def create_lobbies(self, pool: PlayerPool, lobby_size: int, num_lobbies: int) -> List[List[PersistentPlayer]]:
        groups = pool.get_players_by_classification()
        
        # Sort each group randomly
        for g in groups.values():
            np.random.shuffle(g)
        
        lobbies = []
        passive_idx = 0
        aggressive_idx = 0
        neutral_idx = 0
        
        for _ in range(num_lobbies):
            lobby = []
            
            # Try to fill with one type first
            if passive_idx + lobby_size <= len(groups['passive']):
                lobby = groups['passive'][passive_idx:passive_idx + lobby_size]
                passive_idx += lobby_size
            elif aggressive_idx + lobby_size <= len(groups['aggressive']):
                lobby = groups['aggressive'][aggressive_idx:aggressive_idx + lobby_size]
                aggressive_idx += lobby_size
            else:
                # Mix neutrals with remaining
                remaining_passive = groups['passive'][passive_idx:]
                remaining_aggressive = groups['aggressive'][aggressive_idx:]
                remaining_neutral = groups['neutral'][neutral_idx:]
                
                all_remaining = remaining_passive + remaining_aggressive + remaining_neutral
                np.random.shuffle(all_remaining)
                
                if len(all_remaining) >= lobby_size:
                    lobby = all_remaining[:lobby_size]
                    # Update indices (approximate)
                    passive_idx = len(groups['passive'])
                    aggressive_idx = len(groups['aggressive'])
                    neutral_idx = len(groups['neutral'])
            
            if len(lobby) == lobby_size:
                lobbies.append(lobby)
        
        return lobbies


class SBMMMatchmaker(Matchmaker):
    """
    Skill-Based Matchmaking - group similar aggression levels.
    """
    
    @property
    def name(self) -> str:
        return "SBMM"
    
    def create_lobbies(self, pool: PlayerPool, lobby_size: int, num_lobbies: int) -> List[List[PersistentPlayer]]:
        all_players = sorted(pool.get_all_players(), key=lambda p: p.running_aggression)
        
        lobbies = []
        for i in range(num_lobbies):
            start = i * lobby_size
            end = start + lobby_size
            if end <= len(all_players):
                lobbies.append(all_players[start:end])
        
        return lobbies


class DiverseMatchmaker(Matchmaker):
    """
    Force diversity - mix passive, neutral, aggressive.
    Prioritize clear behaviors, fill with neutrals.
    """
    
    @property
    def name(self) -> str:
        return "Diverse"
    
    def create_lobbies(self, pool: PlayerPool, lobby_size: int, num_lobbies: int) -> List[List[PersistentPlayer]]:
        groups = pool.get_players_by_classification()
        
        # Shuffle each group
        for g in groups.values():
            np.random.shuffle(g)
        
        passive_list = list(groups['passive'])
        aggressive_list = list(groups['aggressive'])
        neutral_list = list(groups['neutral'])
        
        lobbies = []
        p_idx, a_idx, n_idx = 0, 0, 0
        
        for _ in range(num_lobbies):
            lobby = []
            
            # Take from passive (target: 4 players)
            take_passive = min(4, len(passive_list) - p_idx, lobby_size - len(lobby))
            lobby.extend(passive_list[p_idx:p_idx + take_passive])
            p_idx += take_passive
            
            # Take from aggressive (target: 4 players)
            take_aggressive = min(4, len(aggressive_list) - a_idx, lobby_size - len(lobby))
            lobby.extend(aggressive_list[a_idx:a_idx + take_aggressive])
            a_idx += take_aggressive
            
            # Fill remaining with neutrals
            remaining = lobby_size - len(lobby)
            take_neutral = min(remaining, len(neutral_list) - n_idx)
            lobby.extend(neutral_list[n_idx:n_idx + take_neutral])
            n_idx += take_neutral
            
            # If still not full, take from any remaining
            if len(lobby) < lobby_size:
                all_remaining = (passive_list[p_idx:] + aggressive_list[a_idx:] + 
                                neutral_list[n_idx:])
                needed = lobby_size - len(lobby)
                if len(all_remaining) >= needed:
                    lobby.extend(all_remaining[:needed])
            
            if len(lobby) == lobby_size:
                lobbies.append(lobby)
        
        return lobbies


# =============================================================================
# EPISODE RUNNER
# =============================================================================

@dataclass
class EpisodeResult:
    """Results from running one episode."""
    matchmaker_name: str
    num_lobbies: int
    lobby_size: int
    
    # Aggregate metrics
    avg_pareto_reward: float
    avg_passive_score: float
    avg_aggressive_score: float
    
    # Detailed per-lobby metrics
    lobby_rewards: List[float]
    lobby_compositions: List[dict]  # {passive, neutral, aggressive counts}
    
    # Player-level stats
    total_extractions: int
    total_deaths: int
    total_kills: int
    
    def to_dict(self) -> dict:
        return {
            'matchmaker': self.matchmaker_name,
            'num_lobbies': self.num_lobbies,
            'lobby_size': self.lobby_size,
            'avg_pareto_reward': self.avg_pareto_reward,
            'avg_passive_score': self.avg_passive_score,
            'avg_aggressive_score': self.avg_aggressive_score,
            'lobby_rewards': self.lobby_rewards,
            'total_extractions': self.total_extractions,
            'total_deaths': self.total_deaths,
            'total_kills': self.total_kills,
        }


class EpisodeRunner:
    """Runs complete episodes with a matchmaker."""
    
    def __init__(self, raid_runner: RaidRunner, num_repeats: int = 10, noise_std: float = 0.05):
        self.raid_runner = raid_runner
        self.num_repeats = num_repeats
        self.noise_std = noise_std
    
    def run_episode(self, pool: PlayerPool, matchmaker: Matchmaker,
                    lobby_size: int = 12, num_lobbies: int = 10,
                    update_players: bool = True) -> EpisodeResult:
        """
        Run one episode:
        1. Matchmaker creates lobbies
        2. Simulate all raids
        3. Compute rewards
        4. Optionally update player stats
        """
        # Create lobbies
        lobbies = matchmaker.create_lobbies(pool, lobby_size, num_lobbies)
        
        if len(lobbies) < num_lobbies:
            print(f"Warning: Only created {len(lobbies)} lobbies (requested {num_lobbies})")
        
        # Build classification map
        classifications = {p.id: p.classification for p in pool.get_all_players()}
        
        # Run raids and collect results
        all_rewards = []
        all_compositions = []
        total_extractions = 0
        total_deaths = 0
        total_kills = 0
        passive_scores = []
        aggressive_scores = []
        
        for lobby in lobbies:
            # Run averaged raid
            results = self.raid_runner.run_averaged_raid(lobby, self.num_repeats, self.noise_std)
            
            # Convert to list format for reward function
            results_list = [
                {'persistent_id': pid, **data}
                for pid, data in results.items()
            ]
            
            # Compute reward
            reward_info = compute_lobby_reward(results_list, classifications)
            all_rewards.append(reward_info['pareto_reward'])
            passive_scores.append(reward_info['passive_score'])
            aggressive_scores.append(reward_info['aggressive_score'])
            
            all_compositions.append({
                'passive': reward_info['passive_count'],
                'neutral': reward_info['neutral_count'],
                'aggressive': reward_info['aggressive_count'],
            })
            
            # Aggregate stats
            for pid, data in results.items():
                if data['extracted']:
                    total_extractions += 1
                else:
                    total_deaths += 1
                total_kills += int(round(data['kills']))
            
            # Update player stats if requested
            if update_players:
                for pp in lobby:
                    r = results[pp.id]
                    pp.record_raid(
                        r['extracted'], r['stash'], r['damage_dealt'],
                        r['damage_received'], int(round(r['kills'])), r['aggression_used']
                    )
                    pp.update_aggression(
                        r['extracted'], int(round(r['kills'])),
                        r['damage_dealt'], r['damage_received'], r['aggression_used']
                    )
        
        return EpisodeResult(
            matchmaker_name=matchmaker.name,
            num_lobbies=len(lobbies),
            lobby_size=lobby_size,
            avg_pareto_reward=round(np.mean(all_rewards), 3) if all_rewards else 0.0,
            avg_passive_score=round(np.mean(passive_scores), 3) if passive_scores else 0.0,
            avg_aggressive_score=round(np.mean(aggressive_scores), 3) if aggressive_scores else 0.0,
            lobby_rewards=all_rewards,
            lobby_compositions=all_compositions,
            total_extractions=total_extractions,
            total_deaths=total_deaths,
            total_kills=total_kills,
        )


# =============================================================================
# EVALUATION FRAMEWORK
# =============================================================================

class Evaluator:
    """Compare matchmaking strategies."""
    
    def __init__(self, pool: PlayerPool, raid_runner: RaidRunner = None,
                 num_repeats: int = 10, noise_std: float = 0.05):
        self.base_pool = pool
        self.raid_runner = raid_runner or RaidRunner()
        self.episode_runner = EpisodeRunner(self.raid_runner, num_repeats, noise_std)
    
    def evaluate_matchmaker(self, matchmaker: Matchmaker, 
                            num_episodes: int = 5,
                            lobby_size: int = 12,
                            num_lobbies: int = 10) -> Dict[str, float]:
        """Evaluate a matchmaker over multiple episodes."""
        # Use copy of pool to not affect original
        pool = self.base_pool.copy()
        
        all_results = []
        for ep in range(num_episodes):
            result = self.episode_runner.run_episode(
                pool, matchmaker, lobby_size, num_lobbies, update_players=True
            )
            all_results.append(result)
        
        # Aggregate across episodes
        return {
            'matchmaker': matchmaker.name,
            'avg_pareto_reward': round(np.mean([r.avg_pareto_reward for r in all_results]), 3),
            'std_pareto_reward': round(np.std([r.avg_pareto_reward for r in all_results]), 3),
            'avg_passive_score': round(np.mean([r.avg_passive_score for r in all_results]), 3),
            'avg_aggressive_score': round(np.mean([r.avg_aggressive_score for r in all_results]), 3),
            'total_extractions': sum(r.total_extractions for r in all_results),
            'total_deaths': sum(r.total_deaths for r in all_results),
            'total_kills': sum(r.total_kills for r in all_results),
        }
    
    def compare_matchmakers(self, matchmakers: List[Matchmaker],
                            num_episodes: int = 5,
                            lobby_size: int = 12,
                            num_lobbies: int = 10,
                            verbose: bool = True) -> List[Dict[str, float]]:
        """Compare multiple matchmakers."""
        results = []
        
        for mm in matchmakers:
            if verbose:
                print(f"Evaluating {mm.name}...")
            
            eval_result = self.evaluate_matchmaker(mm, num_episodes, lobby_size, num_lobbies)
            results.append(eval_result)
            
            if verbose:
                print(f"  Pareto: {eval_result['avg_pareto_reward']:.3f} ± {eval_result['std_pareto_reward']:.3f}")
                print(f"  Passive: {eval_result['avg_passive_score']:.3f}, Aggressive: {eval_result['avg_aggressive_score']:.3f}")
        
        return results


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_comparison(results: List[Dict[str, float]], save_path: str = None):
    """Plot comparison of matchmakers."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    names = [r['matchmaker'] for r in results]
    pareto = [r['avg_pareto_reward'] for r in results]
    passive = [r['avg_passive_score'] for r in results]
    aggressive = [r['avg_aggressive_score'] for r in results]
    
    # Pareto reward
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))
    axes[0].bar(names, pareto, color=colors)
    axes[0].set_ylabel('Pareto Reward')
    axes[0].set_title('Overall Balance (Higher = Better)')
    axes[0].set_ylim(0, max(pareto) * 1.2 if max(pareto) > 0 else 1)
    
    # Passive vs Aggressive scores
    x = np.arange(len(names))
    width = 0.35
    axes[1].bar(x - width/2, passive, width, label='Passive Score', color='green', alpha=0.7)
    axes[1].bar(x + width/2, aggressive, width, label='Aggressive Score', color='red', alpha=0.7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names)
    axes[1].set_ylabel('Score')
    axes[1].set_title('Playstyle Success')
    axes[1].legend()
    
    # Scatter: passive vs aggressive
    axes[2].scatter(passive, aggressive, c=colors, s=200)
    for i, name in enumerate(names):
        axes[2].annotate(name, (passive[i], aggressive[i]), 
                        textcoords="offset points", xytext=(5, 5))
    axes[2].set_xlabel('Passive Score')
    axes[2].set_ylabel('Aggressive Score')
    axes[2].set_title('Pareto Frontier')
    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(0, 1)
    # Draw diagonal (perfect balance)
    axes[2].plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect Balance')
    axes[2].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
    
    plt.close()
    return fig


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def initialize_pool_with_history(pool: PlayerPool, raid_runner: RaidRunner,
                                  target_raids: int = 10, verbose: bool = True) -> int:
    """Initialize pool so all players have history."""
    rounds = 0
    
    while True:
        need_more = [p for p in pool.get_all_players() if p.total_raids < target_raids]
        if not need_more:
            break
        
        need_more.sort(key=lambda p: p.total_raids)
        
        # Select 12 players prioritizing those needing history
        if len(need_more) >= 12:
            selected = need_more[:12]
        else:
            others = [p for p in pool.get_all_players() if p not in need_more]
            np.random.shuffle(others)
            selected = need_more + others[:12 - len(need_more)]
        
        results = raid_runner.run_averaged_raid(selected, num_repeats=10)
        
        for pp in selected:
            r = results[pp.id]
            pp.record_raid(r['extracted'], r['stash'], r['damage_dealt'],
                          r['damage_received'], int(round(r['kills'])), r['aggression_used'])
            pp.update_aggression(r['extracted'], int(round(r['kills'])),
                                r['damage_dealt'], r['damage_received'], r['aggression_used'])
        
        rounds += 1
        
        if verbose and rounds % 20 == 0:
            mn = min(p.total_raids for p in pool.get_all_players())
            mx = max(p.total_raids for p in pool.get_all_players())
            print(f"  Round {rounds}: raids [{mn}, {mx}]")
    
    return rounds


def main():
    """Main entry point - run full evaluation."""
    print("=" * 70)
    print("EXTRACTION SHOOTER MATCHMAKING EVALUATION")
    print("=" * 70)
    
    # Configuration
    NUM_PLAYERS = 1200
    TARGET_HISTORY = 10
    NUM_EPISODES = 5
    LOBBY_SIZE = 12
    NUM_LOBBIES = 10  # Per episode, per matchmaker
    
    # Initialize
    print(f"\n[1] Creating player pool ({NUM_PLAYERS} players, diverse aggression)...")
    np.random.seed(42)
    pool = PlayerPool(NUM_PLAYERS, diverse=True)
    
    # Show initial distribution
    aggrs = [p.aggression for p in pool.get_all_players()]
    print(f"    Initial aggression: mean={np.mean(aggrs):.3f}, std={np.std(aggrs):.3f}")
    print(f"    Range: [{np.min(aggrs):.3f}, {np.max(aggrs):.3f}]")
    initial_groups = pool.get_players_by_classification()
    print(f"    Initial: {len(initial_groups['passive'])} passive, "
          f"{len(initial_groups['neutral'])} neutral, "
          f"{len(initial_groups['aggressive'])} aggressive")
    
    print(f"\n[2] Initializing player histories ({TARGET_HISTORY} raids each)...")
    raid_runner = RaidRunner()
    rounds = initialize_pool_with_history(pool, raid_runner, TARGET_HISTORY, verbose=True)
    print(f"    Completed in {rounds} rounds")
    
    # Save initialized pool
    pool.save('initialized_pool.json')
    print(f"    Saved to initialized_pool.json")
    
    # Pool stats after initialization
    print(f"\n[3] Pool Statistics After Initialization:")
    stats = pool.get_stats()
    for k, v in stats.items():
        print(f"    {k}: {v}")
    
    # Show aggression evolution
    aggrs_after = [p.running_aggression for p in pool.get_all_players()]
    print(f"\n    Aggression evolution:")
    print(f"      Before: mean={np.mean(aggrs):.3f}, std={np.std(aggrs):.3f}")
    print(f"      After:  mean={np.mean(aggrs_after):.3f}, std={np.std(aggrs_after):.3f}")
    
    # Create matchmakers
    print(f"\n[4] Creating matchmakers...")
    matchmakers = [
        RandomMatchmaker(),
        PolarizedMatchmaker(),
        SBMMMatchmaker(),
        DiverseMatchmaker(),
    ]
    print(f"    Created: {[m.name for m in matchmakers]}")
    
    # Evaluate
    print(f"\n[5] Evaluating matchmakers ({NUM_EPISODES} episodes each)...")
    print("-" * 70)
    
    evaluator = Evaluator(pool, raid_runner)
    results = evaluator.compare_matchmakers(
        matchmakers, NUM_EPISODES, LOBBY_SIZE, NUM_LOBBIES, verbose=True
    )
    
    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'Matchmaker':<15} {'Pareto':<12} {'Passive':<12} {'Aggressive':<12}")
    print("-" * 51)
    for r in sorted(results, key=lambda x: -x['avg_pareto_reward']):
        print(f"{r['matchmaker']:<15} {r['avg_pareto_reward']:<12.3f} "
              f"{r['avg_passive_score']:<12.3f} {r['avg_aggressive_score']:<12.3f}")
    
    # Plot
    print(f"\n[6] Creating visualization...")
    plot_comparison(results, 'matchmaker_comparison.png')
    
    # Save results
    print(f"[7] Saving results...")
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("    Saved to evaluation_results.json")
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    
    return pool, results


if __name__ == "__main__":
    pool, results = main()