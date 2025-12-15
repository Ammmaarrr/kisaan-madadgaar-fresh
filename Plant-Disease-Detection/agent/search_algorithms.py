"""
Search Algorithms for Plant Disease Detection

This module implements search algorithms for treatment recommendation
and feature selection optimization.

Algorithms:
1. A* Search: For optimal treatment plan recommendation
2. Genetic Algorithm: For feature selection optimization
"""

import heapq
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TreatmentNode:
    """Represents a state in the treatment search space."""
    disease: str
    severity: float
    status: str  # 'infected', 'improving', 'cured'
    cost: float = 0.0
    path: List[str] = None
    
    def __post_init__(self):
        if self.path is None:
            self.path = []
    
    def __lt__(self, other):
        """For priority queue comparison."""
        return self.cost < other.cost


class TreatmentSearchAStar:
    """
    A* Search algorithm for finding optimal treatment plans.
    
    Problem Formulation:
    - State: (disease, severity, treatment_status)
    - Initial State: (disease_name, severity=100, status='infected')
    - Goal State: severity=0, status='cured'
    - Actions: Apply treatments
    - Path Cost: Treatment cost + time
    - Heuristic: Estimated remaining treatment cost based on severity
    
    The algorithm finds the optimal sequence of treatments that:
    1. Minimizes total cost (monetary + time)
    2. Maximizes treatment effectiveness
    3. Considers resource availability
    """
    
    def __init__(self, treatment_graph: Dict[str, List[Dict]]):
        """
        Initialize A* search with treatment knowledge base.
        
        Args:
            treatment_graph (dict): Treatment options for each disease
                Format: {
                    'disease_name': [
                        {
                            'treatment': 'treatment_name',
                            'cost': 50,
                            'effectiveness': 80,
                            'time': 7  # days
                        },
                        ...
                    ]
                }
        """
        self.graph = treatment_graph
        self.nodes_expanded = 0
        self.max_frontier_size = 0
        logger.info(f"TreatmentSearchAStar initialized with {len(treatment_graph)} diseases")
    
    def heuristic(self, current_state: TreatmentNode, goal_state: TreatmentNode) -> float:
        """
        Admissible heuristic function for A* search.
        
        Estimates the minimum cost to reach the goal from current state.
        
        Args:
            current_state (TreatmentNode): Current state
            goal_state (TreatmentNode): Goal state
            
        Returns:
            float: Estimated cost to goal (admissible - never overestimates)
        """
        if current_state.status == 'cured':
            return 0.0
        
        # Heuristic: severity proportional to minimum treatment cost
        # Assume best-case scenario: most effective treatment
        severity_factor = current_state.severity / 100.0
        
        # Get minimum cost treatment for this disease
        treatments = self.graph.get(current_state.disease, [])
        if not treatments:
            return severity_factor * 100  # Default high cost
        
        min_cost = min(t['cost'] for t in treatments)
        max_effectiveness = max(t['effectiveness'] for t in treatments)
        
        # Estimate: (remaining severity / max effectiveness) * min cost
        estimated_treatments_needed = severity_factor * (100 / max_effectiveness)
        estimated_cost = estimated_treatments_needed * min_cost
        
        return estimated_cost
    
    def search(self, disease: str, initial_severity: float = 100.0,
               available_resources: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Perform A* search to find optimal treatment plan.
        
        Args:
            disease (str): Disease name
            initial_severity (float): Initial disease severity (0-100)
            available_resources (dict): Available resources/constraints
            
        Returns:
            dict: Optimal treatment plan with path, cost, and effectiveness
        """
        if disease not in self.graph:
            logger.warning(f"Disease '{disease}' not found in treatment database")
            return {
                'success': False,
                'message': f"No treatment data available for {disease}",
                'disease': disease
            }
        
        # Initialize search
        self.nodes_expanded = 0
        start_state = TreatmentNode(
            disease=disease,
            severity=initial_severity,
            status='infected',
            cost=0.0,
            path=[]
        )
        
        goal_state = TreatmentNode(
            disease=disease,
            severity=0.0,
            status='cured'
        )
        
        # Priority queue: (f_cost, node)
        # f_cost = g_cost + h_cost
        h_start = self.heuristic(start_state, goal_state)
        frontier = [(h_start, 0, start_state)]  # (f_cost, g_cost, node)
        
        # Track explored states
        explored = set()
        
        # Search
        while frontier:
            self.max_frontier_size = max(self.max_frontier_size, len(frontier))
            
            f_cost, g_cost, current = heapq.heappop(frontier)
            
            # Goal test
            if current.severity <= 0 or current.status == 'cured':
                logger.info(f"Solution found! Nodes expanded: {self.nodes_expanded}")
                return self._format_solution(current, g_cost)
            
            # Mark as explored
            state_key = (current.disease, round(current.severity, 2), current.status)
            if state_key in explored:
                continue
            explored.add(state_key)
            
            self.nodes_expanded += 1
            
            # Expand node: try all available treatments
            treatments = self.graph.get(current.disease, [])
            
            for treatment_info in treatments:
                # Check resource constraints
                if available_resources and not self._check_resources(
                    treatment_info, available_resources
                ):
                    continue
                
                # Generate successor state
                new_severity = max(0, current.severity - treatment_info['effectiveness'])
                new_status = 'cured' if new_severity == 0 else 'improving'
                
                new_path = current.path + [treatment_info['treatment']]
                new_g_cost = g_cost + treatment_info['cost']
                
                new_node = TreatmentNode(
                    disease=current.disease,
                    severity=new_severity,
                    status=new_status,
                    cost=new_g_cost,
                    path=new_path
                )
                
                # Calculate f_cost
                h_cost = self.heuristic(new_node, goal_state)
                new_f_cost = new_g_cost + h_cost
                
                # Add to frontier
                heapq.heappush(frontier, (new_f_cost, new_g_cost, new_node))
        
        # No solution found
        logger.warning(f"No solution found for {disease}")
        return {
            'success': False,
            'message': 'No treatment plan found',
            'disease': disease,
            'nodes_expanded': self.nodes_expanded
        }
    
    def _format_solution(self, final_node: TreatmentNode, total_cost: float) -> Dict[str, Any]:
        """Format the solution for output."""
        return {
            'success': True,
            'disease': final_node.disease,
            'treatment_plan': final_node.path,
            'total_cost': total_cost,
            'num_treatments': len(final_node.path),
            'nodes_expanded': self.nodes_expanded,
            'max_frontier_size': self.max_frontier_size,
            'status': 'Optimal treatment plan found',
            'detailed_plan': self._get_detailed_plan(final_node)
        }
    
    def _get_detailed_plan(self, final_node: TreatmentNode) -> List[Dict]:
        """Get detailed treatment plan with costs and effectiveness."""
        detailed = []
        treatments = self.graph.get(final_node.disease, [])
        
        for treatment_name in final_node.path:
            # Find treatment details
            treatment_info = next(
                (t for t in treatments if t['treatment'] == treatment_name),
                None
            )
            if treatment_info:
                detailed.append({
                    'treatment': treatment_name,
                    'cost': treatment_info['cost'],
                    'effectiveness': treatment_info['effectiveness'],
                    'time_days': treatment_info.get('time', 'N/A')
                })
        
        return detailed
    
    def _check_resources(self, treatment: Dict, available_resources: Dict) -> bool:
        """Check if treatment is feasible given available resources."""
        # Check budget constraint
        if 'budget' in available_resources:
            if treatment['cost'] > available_resources['budget']:
                return False
        
        # Check time constraint
        if 'max_time' in available_resources and 'time' in treatment:
            if treatment['time'] > available_resources['max_time']:
                return False
        
        return True
    
    def get_statistics(self) -> Dict[str, int]:
        """Get search statistics."""
        return {
            'nodes_expanded': self.nodes_expanded,
            'max_frontier_size': self.max_frontier_size
        }


class FeatureSelectionGA:
    """
    Genetic Algorithm for optimal feature selection.
    
    Problem: Select the best subset of image features for disease classification
    
    Representation: Binary chromosome [1,0,1,1,0...] where 1=selected, 0=not selected
    Fitness: Model accuracy - (penalty * number of features)
    Selection: Tournament selection
    Crossover: Single-point crossover
    Mutation: Bit flip
    """
    
    def __init__(self, 
                 population_size: int = 50,
                 generations: int = 100,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.1,
                 tournament_size: int = 5,
                 feature_penalty: float = 0.001):
        """
        Initialize Genetic Algorithm.
        
        Args:
            population_size (int): Number of individuals in population
            generations (int): Number of generations to evolve
            crossover_rate (float): Probability of crossover (0-1)
            mutation_rate (float): Probability of mutation per bit (0-1)
            tournament_size (int): Size of tournament for selection
            feature_penalty (float): Penalty for using more features
        """
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.feature_penalty = feature_penalty
        
        self.population = None
        self.fitness_scores = None
        self.best_individual = None
        self.best_fitness = -np.inf
        self.fitness_history = []
        
        logger.info(f"FeatureSelectionGA initialized: pop_size={population_size}, "
                   f"generations={generations}")
    
    def initialize_population(self, num_features: int) -> np.ndarray:
        """
        Initialize random population.
        
        Args:
            num_features (int): Total number of features
            
        Returns:
            np.ndarray: Population matrix (population_size x num_features)
        """
        # Initialize with at least some features selected (0.3 probability)
        self.population = np.random.binomial(1, 0.3, 
                                            (self.population_size, num_features))
        
        # Ensure at least 1 feature is selected in each individual
        for i in range(self.population_size):
            if np.sum(self.population[i]) == 0:
                # Randomly select a few features
                num_to_select = np.random.randint(1, max(2, num_features // 10))
                selected_indices = np.random.choice(num_features, num_to_select, 
                                                   replace=False)
                self.population[i, selected_indices] = 1
        
        logger.info(f"Population initialized with {num_features} features")
        return self.population
    
    def fitness_function(self, individual: np.ndarray, X: np.ndarray, 
                        y: np.ndarray, model_evaluator) -> float:
        """
        Evaluate fitness of an individual.
        
        Fitness = Model Accuracy - (Feature Penalty * Number of Selected Features)
        
        Args:
            individual (np.ndarray): Binary feature selection array
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target labels
            model_evaluator: Function that trains and evaluates model
            
        Returns:
            float: Fitness score
        """
        # Get selected features
        selected_features = np.where(individual == 1)[0]
        
        if len(selected_features) == 0:
            return 0.0  # Invalid individual
        
        # Select features from X
        X_selected = X[:, selected_features]
        
        # Evaluate model with selected features
        accuracy = model_evaluator(X_selected, y)
        
        # Calculate fitness with feature penalty
        num_features = len(selected_features)
        fitness = accuracy - (self.feature_penalty * num_features)
        
        return fitness
    
    def evaluate_population(self, X: np.ndarray, y: np.ndarray, 
                           model_evaluator) -> np.ndarray:
        """
        Evaluate fitness for entire population.
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target labels
            model_evaluator: Function that trains and evaluates model
            
        Returns:
            np.ndarray: Fitness scores for population
        """
        fitness_scores = np.zeros(self.population_size)
        
        for i in range(self.population_size):
            fitness_scores[i] = self.fitness_function(
                self.population[i], X, y, model_evaluator
            )
        
        self.fitness_scores = fitness_scores
        
        # Track best individual
        best_idx = np.argmax(fitness_scores)
        if fitness_scores[best_idx] > self.best_fitness:
            self.best_fitness = fitness_scores[best_idx]
            self.best_individual = self.population[best_idx].copy()
        
        return fitness_scores
    
    def tournament_selection(self) -> np.ndarray:
        """
        Tournament selection: Select best from random tournament.
        
        Returns:
            np.ndarray: Selected individual
        """
        tournament_indices = np.random.choice(
            self.population_size, 
            self.tournament_size, 
            replace=False
        )
        tournament_fitness = self.fitness_scores[tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        
        return self.population[winner_idx].copy()
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Single-point crossover.
        
        Args:
            parent1 (np.ndarray): First parent
            parent2 (np.ndarray): Second parent
            
        Returns:
            tuple: Two offspring
        """
        if np.random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        # Single-point crossover
        crossover_point = np.random.randint(1, len(parent1))
        
        offspring1 = np.concatenate([
            parent1[:crossover_point],
            parent2[crossover_point:]
        ])
        
        offspring2 = np.concatenate([
            parent2[:crossover_point],
            parent1[crossover_point:]
        ])
        
        return offspring1, offspring2
    
    def mutate(self, individual: np.ndarray) -> np.ndarray:
        """
        Bit-flip mutation.
        
        Args:
            individual (np.ndarray): Individual to mutate
            
        Returns:
            np.ndarray: Mutated individual
        """
        mutated = individual.copy()
        
        for i in range(len(mutated)):
            if np.random.random() < self.mutation_rate:
                mutated[i] = 1 - mutated[i]  # Flip bit
        
        # Ensure at least one feature is selected
        if np.sum(mutated) == 0:
            random_feature = np.random.randint(0, len(mutated))
            mutated[random_feature] = 1
        
        return mutated
    
    def evolve(self, X: np.ndarray, y: np.ndarray, model_evaluator) -> Dict[str, Any]:
        """
        Run the genetic algorithm.
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target labels
            model_evaluator: Function that trains and evaluates model
            
        Returns:
            dict: Results with best individual and fitness history
        """
        num_features = X.shape[1]
        
        # Initialize population
        self.initialize_population(num_features)
        
        logger.info(f"Starting evolution for {self.generations} generations...")
        
        for generation in range(self.generations):
            # Evaluate population
            fitness_scores = self.evaluate_population(X, y, model_evaluator)
            
            # Track statistics
            avg_fitness = np.mean(fitness_scores)
            max_fitness = np.max(fitness_scores)
            self.fitness_history.append({
                'generation': generation,
                'avg_fitness': avg_fitness,
                'max_fitness': max_fitness,
                'best_fitness': self.best_fitness
            })
            
            if generation % 10 == 0:
                logger.info(f"Generation {generation}: "
                          f"Avg Fitness={avg_fitness:.4f}, "
                          f"Max Fitness={max_fitness:.4f}")
            
            # Create new population
            new_population = []
            
            # Elitism: Keep best individual
            new_population.append(self.best_individual.copy())
            
            # Generate rest of population
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self.tournament_selection()
                parent2 = self.tournament_selection()
                
                # Crossover
                offspring1, offspring2 = self.crossover(parent1, parent2)
                
                # Mutation
                offspring1 = self.mutate(offspring1)
                offspring2 = self.mutate(offspring2)
                
                new_population.append(offspring1)
                if len(new_population) < self.population_size:
                    new_population.append(offspring2)
            
            self.population = np.array(new_population[:self.population_size])
        
        # Final evaluation
        self.evaluate_population(X, y, model_evaluator)
        
        logger.info(f"Evolution complete. Best fitness: {self.best_fitness:.4f}")
        
        return self._format_results()
    
    def _format_results(self) -> Dict[str, Any]:
        """Format GA results."""
        selected_features = np.where(self.best_individual == 1)[0]
        
        return {
            'success': True,
            'best_individual': self.best_individual,
            'selected_features': selected_features.tolist(),
            'num_selected_features': len(selected_features),
            'best_fitness': self.best_fitness,
            'fitness_history': self.fitness_history,
            'final_population': self.population,
            'generations_completed': self.generations
        }
    
    def get_best_features(self) -> np.ndarray:
        """
        Get indices of selected features from best individual.
        
        Returns:
            np.ndarray: Indices of selected features
        """
        if self.best_individual is None:
            return np.array([])
        
        return np.where(self.best_individual == 1)[0]
