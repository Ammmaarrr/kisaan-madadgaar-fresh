"""
Treatment Database Management

This module manages the treatment knowledge base for plant diseases,
providing interface for the A* search algorithm.
"""

import json
import os
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TreatmentDatabase:
    """
    Treatment knowledge base for plant diseases.
    
    Stores treatment options with:
    - Treatment name
    - Cost (monetary)
    - Effectiveness (0-100)
    - Time required (days)
    - Prerequisites
    - Side effects/considerations
    """
    
    def __init__(self, database_path: Optional[str] = None):
        """
        Initialize treatment database.
        
        Args:
            database_path (str, optional): Path to JSON database file
        """
        self.database = {}
        self.database_path = database_path
        
        if database_path and os.path.exists(database_path):
            self.load_from_file(database_path)
        else:
            self._initialize_default_database()
        
        logger.info(f"TreatmentDatabase initialized with {len(self.database)} diseases")
    
    def _initialize_default_database(self):
        """Initialize with default treatment knowledge."""
        self.database = {
            "Apple___Apple_scab": [
                {
                    "treatment": "Fungicide_spray_Captan",
                    "cost": 50,
                    "effectiveness": 80,
                    "time": 14,
                    "application": "Spray every 7-10 days",
                    "notes": "Most effective when applied preventively"
                },
                {
                    "treatment": "Remove_infected_leaves",
                    "cost": 10,
                    "effectiveness": 40,
                    "time": 3,
                    "application": "Manual removal and disposal",
                    "notes": "Reduces disease spread"
                },
                {
                    "treatment": "Copper_based_treatment",
                    "cost": 70,
                    "effectiveness": 90,
                    "time": 21,
                    "application": "Apply during dormant season",
                    "notes": "Highly effective but more expensive"
                },
                {
                    "treatment": "Sulfur_dust",
                    "cost": 30,
                    "effectiveness": 60,
                    "time": 10,
                    "application": "Dust foliage thoroughly",
                    "notes": "Organic option"
                }
            ],
            "Apple___Black_rot": [
                {
                    "treatment": "Fungicide_Thiophanate_methyl",
                    "cost": 60,
                    "effectiveness": 85,
                    "time": 14,
                    "application": "Apply at petal fall",
                    "notes": "Systemic fungicide"
                },
                {
                    "treatment": "Prune_infected_areas",
                    "cost": 20,
                    "effectiveness": 50,
                    "time": 5,
                    "application": "Cut 6-8 inches below visible infection",
                    "notes": "Essential first step"
                },
                {
                    "treatment": "Lime_sulfur_spray",
                    "cost": 40,
                    "effectiveness": 70,
                    "time": 14,
                    "application": "Dormant season application",
                    "notes": "Preventive measure"
                }
            ],
            "Tomato___Bacterial_spot": [
                {
                    "treatment": "Copper_spray",
                    "cost": 60,
                    "effectiveness": 75,
                    "time": 14,
                    "application": "Spray weekly",
                    "notes": "Weather dependent effectiveness"
                },
                {
                    "treatment": "Crop_rotation",
                    "cost": 100,
                    "effectiveness": 60,
                    "time": 365,
                    "application": "Rotate with non-host crops",
                    "notes": "Long-term prevention"
                },
                {
                    "treatment": "Remove_infected_plants",
                    "cost": 15,
                    "effectiveness": 45,
                    "time": 1,
                    "application": "Immediate removal",
                    "notes": "Prevents spread"
                },
                {
                    "treatment": "Bactericide_streptomycin",
                    "cost": 80,
                    "effectiveness": 85,
                    "time": 10,
                    "application": "Apply at first sign",
                    "notes": "Most effective early treatment"
                }
            ],
            "Tomato___Early_blight": [
                {
                    "treatment": "Fungicide_Chlorothalonil",
                    "cost": 55,
                    "effectiveness": 80,
                    "time": 14,
                    "application": "Spray every 7-10 days",
                    "notes": "Preventive application recommended"
                },
                {
                    "treatment": "Mulching",
                    "cost": 25,
                    "effectiveness": 40,
                    "time": 30,
                    "application": "Apply organic mulch",
                    "notes": "Reduces soil splash"
                },
                {
                    "treatment": "Remove_lower_leaves",
                    "cost": 10,
                    "effectiveness": 35,
                    "time": 2,
                    "application": "Prune lower 12 inches",
                    "notes": "Improves air circulation"
                },
                {
                    "treatment": "Copper_fungicide",
                    "cost": 65,
                    "effectiveness": 75,
                    "time": 14,
                    "application": "Organic option",
                    "notes": "OMRI listed"
                }
            ],
            "Tomato___Late_blight": [
                {
                    "treatment": "Fungicide_Mancozeb",
                    "cost": 70,
                    "effectiveness": 85,
                    "time": 10,
                    "application": "Spray immediately",
                    "notes": "Aggressive treatment needed"
                },
                {
                    "treatment": "Destroy_infected_plants",
                    "cost": 20,
                    "effectiveness": 60,
                    "time": 1,
                    "application": "Complete removal",
                    "notes": "Critical to prevent spread"
                },
                {
                    "treatment": "Systemic_fungicide_Ridomil",
                    "cost": 90,
                    "effectiveness": 95,
                    "time": 14,
                    "application": "Soil drench and foliar",
                    "notes": "Most effective treatment"
                }
            ],
            "Potato___Early_blight": [
                {
                    "treatment": "Fungicide_Azoxystrobin",
                    "cost": 65,
                    "effectiveness": 85,
                    "time": 14,
                    "application": "Apply at first symptoms",
                    "notes": "Broad spectrum"
                },
                {
                    "treatment": "Hill_soil_around_plants",
                    "cost": 15,
                    "effectiveness": 40,
                    "time": 7,
                    "application": "Cover lower stems",
                    "notes": "Protects tubers"
                },
                {
                    "treatment": "Compost_tea_spray",
                    "cost": 20,
                    "effectiveness": 50,
                    "time": 21,
                    "application": "Weekly application",
                    "notes": "Organic prevention"
                }
            ],
            "Potato___Late_blight": [
                {
                    "treatment": "Fungicide_Cymoxanil",
                    "cost": 75,
                    "effectiveness": 90,
                    "time": 10,
                    "application": "Preventive spray",
                    "notes": "Highly effective"
                },
                {
                    "treatment": "Remove_all_infected_plants",
                    "cost": 30,
                    "effectiveness": 65,
                    "time": 2,
                    "application": "Immediate destruction",
                    "notes": "Essential containment"
                },
                {
                    "treatment": "Copper_hydroxide",
                    "cost": 60,
                    "effectiveness": 75,
                    "time": 14,
                    "application": "Organic alternative",
                    "notes": "Weather dependent"
                }
            ],
            "Corn___Common_rust": [
                {
                    "treatment": "Fungicide_Triazole",
                    "cost": 55,
                    "effectiveness": 80,
                    "time": 14,
                    "application": "Apply at first pustules",
                    "notes": "Economical threshold dependent"
                },
                {
                    "treatment": "Plant_resistant_varieties",
                    "cost": 50,
                    "effectiveness": 70,
                    "time": 120,
                    "application": "Next season prevention",
                    "notes": "Long-term solution"
                },
                {
                    "treatment": "No_treatment_if_mild",
                    "cost": 0,
                    "effectiveness": 30,
                    "time": 0,
                    "application": "Monitor only",
                    "notes": "Often not economically justified"
                }
            ],
            "Grape___Black_rot": [
                {
                    "treatment": "Fungicide_Myclobutanil",
                    "cost": 70,
                    "effectiveness": 90,
                    "time": 21,
                    "application": "Spray from bloom to harvest",
                    "notes": "Multiple applications needed"
                },
                {
                    "treatment": "Remove_mummified_fruit",
                    "cost": 25,
                    "effectiveness": 50,
                    "time": 7,
                    "application": "Sanitation critical",
                    "notes": "Reduces inoculum"
                },
                {
                    "treatment": "Sulfur_spray",
                    "cost": 40,
                    "effectiveness": 65,
                    "time": 21,
                    "application": "Organic option",
                    "notes": "Less effective than synthetics"
                }
            ],
            "Pepper___Bacterial_spot": [
                {
                    "treatment": "Copper_bactericide",
                    "cost": 65,
                    "effectiveness": 70,
                    "time": 14,
                    "application": "Weekly sprays",
                    "notes": "Resistance may develop"
                },
                {
                    "treatment": "Plant_resistant_varieties",
                    "cost": 40,
                    "effectiveness": 80,
                    "time": 90,
                    "application": "Future plantings",
                    "notes": "Best long-term strategy"
                },
                {
                    "treatment": "Remove_infected_leaves",
                    "cost": 15,
                    "effectiveness": 45,
                    "time": 3,
                    "application": "Regular scouting",
                    "notes": "Early intervention"
                }
            ]
        }
        
        logger.info("Default treatment database initialized")
    
    def load_from_file(self, filepath: str) -> None:
        """
        Load treatment database from JSON file.
        
        Args:
            filepath (str): Path to JSON database file
        """
        try:
            with open(filepath, 'r') as f:
                self.database = json.load(f)
            logger.info(f"Treatment database loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading database from {filepath}: {e}")
            self._initialize_default_database()
    
    def save_to_file(self, filepath: str) -> None:
        """
        Save treatment database to JSON file.
        
        Args:
            filepath (str): Path to save JSON database
        """
        try:
            with open(filepath, 'w') as f:
                json.dump(self.database, f, indent=2)
            logger.info(f"Treatment database saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving database to {filepath}: {e}")
    
    def get_treatments(self, disease: str) -> List[Dict]:
        """
        Get all treatment options for a disease.
        
        Args:
            disease (str): Disease name
            
        Returns:
            list: List of treatment dictionaries
        """
        return self.database.get(disease, [])
    
    def add_treatment(self, disease: str, treatment: Dict) -> None:
        """
        Add a new treatment option for a disease.
        
        Args:
            disease (str): Disease name
            treatment (dict): Treatment information
        """
        if disease not in self.database:
            self.database[disease] = []
        
        self.database[disease].append(treatment)
        logger.info(f"Added treatment '{treatment['treatment']}' for {disease}")
    
    def get_database(self) -> Dict:
        """
        Get entire treatment database.
        
        Returns:
            dict: Complete treatment database
        """
        return self.database
    
    def get_disease_list(self) -> List[str]:
        """
        Get list of all diseases in database.
        
        Returns:
            list: Disease names
        """
        return list(self.database.keys())
    
    def search_by_cost(self, disease: str, max_cost: float) -> List[Dict]:
        """
        Get treatments within cost budget.
        
        Args:
            disease (str): Disease name
            max_cost (float): Maximum cost
            
        Returns:
            list: Treatments within budget
        """
        treatments = self.get_treatments(disease)
        return [t for t in treatments if t['cost'] <= max_cost]
    
    def search_by_effectiveness(self, disease: str, min_effectiveness: float) -> List[Dict]:
        """
        Get treatments above effectiveness threshold.
        
        Args:
            disease (str): Disease name
            min_effectiveness (float): Minimum effectiveness (0-100)
            
        Returns:
            list: Effective treatments
        """
        treatments = self.get_treatments(disease)
        return [t for t in treatments if t['effectiveness'] >= min_effectiveness]
    
    def get_best_treatment(self, disease: str, criterion: str = 'effectiveness') -> Optional[Dict]:
        """
        Get best treatment based on criterion.
        
        Args:
            disease (str): Disease name
            criterion (str): 'effectiveness', 'cost', or 'time'
            
        Returns:
            dict: Best treatment option
        """
        treatments = self.get_treatments(disease)
        
        if not treatments:
            return None
        
        if criterion == 'effectiveness':
            return max(treatments, key=lambda x: x['effectiveness'])
        elif criterion == 'cost':
            return min(treatments, key=lambda x: x['cost'])
        elif criterion == 'time':
            return min(treatments, key=lambda x: x['time'])
        else:
            return treatments[0]
    
    def get_statistics(self) -> Dict:
        """
        Get database statistics.
        
        Returns:
            dict: Statistics about the database
        """
        total_diseases = len(self.database)
        total_treatments = sum(len(treatments) for treatments in self.database.values())
        
        avg_treatments_per_disease = total_treatments / total_diseases if total_diseases > 0 else 0
        
        all_treatments = [t for treatments in self.database.values() for t in treatments]
        avg_cost = sum(t['cost'] for t in all_treatments) / total_treatments if total_treatments > 0 else 0
        avg_effectiveness = sum(t['effectiveness'] for t in all_treatments) / total_treatments if total_treatments > 0 else 0
        avg_time = sum(t['time'] for t in all_treatments) / total_treatments if total_treatments > 0 else 0
        
        return {
            'total_diseases': total_diseases,
            'total_treatments': total_treatments,
            'avg_treatments_per_disease': avg_treatments_per_disease,
            'avg_cost': avg_cost,
            'avg_effectiveness': avg_effectiveness,
            'avg_time_days': avg_time
        }
