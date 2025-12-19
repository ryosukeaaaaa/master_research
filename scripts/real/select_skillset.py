"""
Skill Set Selection for DINA Model Analysis

This module provides three strategies for selecting skill sets that maximize
the number of valid users who solve problems covering all selected skills
in both the first and second halves of their problem sequences.

Strategies:
    A. Frequency-Based: Select skills with highest user coverage
    B. Co-occurrence Maximization: Greedy selection to maximize valid users
    C. Hybrid: Balance frequency, temporal distribution, and dual coverage
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Set, Tuple
from pathlib import Path


class SkillSetSelector:
    """
    Skill set selector for time-series analysis with DINA model.
    
    Attributes:
        df (pd.DataFrame): Dataset with columns: user_id, skill_id, order_id
        df_split (pd.DataFrame): Dataset with timeline split into first/second halves
        coverage_matrix (Dict): Mapping from user_id to set of dual-covered skills
    """
    
    def __init__(self, df: pd.DataFrame, verbose: bool = True):
        """
        Initialize the selector.
        
        Args:
            df: DataFrame with at least columns: user_id, skill_id, order_id
            verbose: Whether to print progress messages
        """
        self.df = df.copy()
        self.verbose = verbose
        
        # Sort by user and time
        if 'order_id' not in self.df.columns:
            # If no order_id, create one based on row order per user
            self.df['order_id'] = self.df.groupby('user_id').cumcount()
        
        self.df = self.df.sort_values(['user_id', 'order_id']).reset_index(drop=True)
        
        # Initialize split data and coverage matrix
        self.df_split = None
        self.coverage_matrix = None
        
        if self.verbose:
            print(f"Initialized with {len(self.df)} records")
            print(f"Users: {self.df['user_id'].nunique()}")
            print(f"Skills: {self.df['skill_id'].nunique()}")
    
    def split_timeline(self) -> pd.DataFrame:
        """
        Split each user's problem sequence into first and second halves.
        
        Returns:
            DataFrame with additional 'half' column ('first' or 'second')
        """
        if self.df_split is not None:
            return self.df_split
        
        def assign_half(group):
            n = len(group)
            mid = n // 2
            group = group.copy()
            group['half'] = ['first'] * mid + ['second'] * (n - mid)
            return group
        
        self.df_split = self.df.groupby('user_id', group_keys=False).apply(assign_half)
        
        if self.verbose:
            print("Timeline split completed")
            print(f"First half records: {(self.df_split['half'] == 'first').sum()}")
            print(f"Second half records: {(self.df_split['half'] == 'second').sum()}")
        
        return self.df_split
    
    def compute_dual_coverage(self, df_split: pd.DataFrame = None) -> Dict[int, Set[float]]:
        """
        Compute which skills each user covers in BOTH first and second halves.
        
        Args:
            df_split: DataFrame with 'half' column. If None, uses self.df_split
        
        Returns:
            Dictionary mapping user_id to set of dual-covered skill_ids
        """
        if df_split is None:
            if self.df_split is None:
                df_split = self.split_timeline()
            else:
                df_split = self.df_split
        
        if self.coverage_matrix is not None:
            return self.coverage_matrix
        
        coverage = {}
        for user in df_split['user_id'].unique():
            user_data = df_split[df_split['user_id'] == user]
            first_skills = set(user_data[user_data['half'] == 'first']['skill_id'].dropna())
            second_skills = set(user_data[user_data['half'] == 'second']['skill_id'].dropna())
            coverage[user] = first_skills & second_skills
        
        self.coverage_matrix = coverage
        
        if self.verbose:
            avg_coverage = np.mean([len(skills) for skills in coverage.values()])
            print(f"Dual coverage computed: avg {avg_coverage:.2f} skills per user")
        
        return coverage
    
    def strategy_frequency(self, K: int, min_users: int = 500) -> List[float]:
        """
        Strategy A: Select K skills with highest user coverage.
        
        Args:
            K: Number of skills to select
            min_users: Minimum number of users who must solve the skill
        
        Returns:
            List of selected skill IDs
        """
        skill_counts = self.df.groupby('skill_id')['user_id'].nunique()
        candidates = skill_counts[skill_counts >= min_users]
        
        if len(candidates) < K:
            print(f"Warning: Only {len(candidates)} skills meet min_users={min_users}")
            K = len(candidates)
        
        selected = candidates.nlargest(K).index.tolist()
        
        if self.verbose:
            print(f"\nStrategy A (Frequency-Based): Selected {len(selected)} skills")
        
        return selected
    
    def strategy_cooccurrence(self, K: int, min_users: int = 500) -> Tuple[List[float], int]:
        """
        Strategy B: Greedy selection to maximize number of valid users.
        
        A user is valid if they cover all selected skills in both halves.
        
        Args:
            K: Number of skills to select
            min_users: Minimum number of users for candidate skills
        
        Returns:
            Tuple of (selected skill IDs, number of valid users)
        """
        # Ensure coverage matrix is computed
        if self.df_split is None:
            self.split_timeline()
        if self.coverage_matrix is None:
            self.compute_dual_coverage()
        
        # Get candidate skills
        skill_counts = self.df.groupby('skill_id')['user_id'].nunique()
        candidates = skill_counts[skill_counts >= min_users].index.tolist()
        
        if len(candidates) < K:
            print(f"Warning: Only {len(candidates)} skills meet min_users={min_users}")
            K = len(candidates)
        
        selected_skills = []
        
        if self.verbose:
            print(f"\nStrategy B (Co-occurrence): Greedy selection for K={K}")
        
        for i in range(K):
            best_skill = None
            best_count = 0
            
            for skill in candidates:
                if skill in selected_skills:
                    continue
                
                # Count valid users if we add this skill
                temp_selected = set(selected_skills + [skill])
                valid_count = sum(1 for user_skills in self.coverage_matrix.values()
                                 if temp_selected.issubset(user_skills))
                
                if valid_count > best_count:
                    best_count = valid_count
                    best_skill = skill
            
            if best_skill is None:
                print(f"No more skills can be added at step {i+1}")
                break
            
            selected_skills.append(best_skill)
            
            if self.verbose:
                print(f"  Step {i+1}: Skill {best_skill} -> {best_count} valid users")
        
        # Final count
        final_valid = sum(1 for user_skills in self.coverage_matrix.values()
                         if set(selected_skills).issubset(user_skills))
        
        return selected_skills, final_valid
    
    def strategy_hybrid(self, K: int, min_users: int = 500, 
                       temporal_weight: float = 0.3) -> List[float]:
        """
        Strategy C: Hybrid approach balancing frequency, temporal balance, and dual coverage.
        
        Args:
            K: Number of skills to select
            min_users: Minimum number of users for candidate skills
            temporal_weight: Weight for temporal balance (0-1)
        
        Returns:
            List of selected skill IDs
        """
        if self.df_split is None:
            self.split_timeline()
        
        skill_counts = self.df.groupby('skill_id')['user_id'].nunique()
        candidates = skill_counts[skill_counts >= min_users].index.tolist()
        
        if len(candidates) < K:
            print(f"Warning: Only {len(candidates)} skills meet min_users={min_users}")
            K = len(candidates)
        
        if self.verbose:
            print(f"\nStrategy C (Hybrid): Computing scores for {len(candidates)} candidates")
        
        skill_scores = {}
        max_freq = skill_counts.max()
        
        for skill in candidates:
            # Frequency score (normalized)
            freq_score = skill_counts[skill] / max_freq
            
            # Temporal balance score
            temporal_score = self._compute_temporal_balance(self.df_split, skill)
            
            # Dual coverage rate
            dual_coverage = self._compute_dual_coverage_rate(self.df_split, skill)
            
            # Combined score
            skill_scores[skill] = (
                (1 - temporal_weight) * freq_score +
                temporal_weight * temporal_score * dual_coverage
            )
        
        # Select top K
        selected = sorted(skill_scores.items(), key=lambda x: x[1], reverse=True)[:K]
        selected_skills = [s[0] for s in selected]
        
        if self.verbose:
            print(f"Selected {len(selected_skills)} skills with scores:")
            for skill, score in selected[:10]:
                print(f"  Skill {skill}: {score:.4f}")
        
        return selected_skills
    
    def _compute_temporal_balance(self, df_split: pd.DataFrame, skill: float) -> float:
        """
        Compute temporal balance: how evenly the skill is distributed across time.
        
        Returns a score between 0 and 1, where 1 means perfectly balanced.
        """
        df_skill = df_split[df_split['skill_id'] == skill]
        balances = []
        
        for user in df_skill['user_id'].unique():
            user_data = df_skill[df_skill['user_id'] == user]
            first_count = (user_data['half'] == 'first').sum()
            second_count = (user_data['half'] == 'second').sum()
            total = first_count + second_count
            
            if total >= 2:
                # Balance score: 1 if equal, 0 if all in one half
                balance = 1 - abs(first_count - second_count) / total
                balances.append(balance)
        
        return np.mean(balances) if balances else 0
    
    def _compute_dual_coverage_rate(self, df_split: pd.DataFrame, skill: float) -> float:
        """
        Compute the proportion of users who solve this skill in both halves.
        """
        df_skill = df_split[df_split['skill_id'] == skill]
        users = df_skill['user_id'].unique()
        
        dual_count = 0
        for user in users:
            user_data = df_skill[df_skill['user_id'] == user]
            has_first = (user_data['half'] == 'first').any()
            has_second = (user_data['half'] == 'second').any()
            if has_first and has_second:
                dual_count += 1
        
        return dual_count / len(users) if len(users) > 0 else 0
    
    def evaluate_skillset(self, skills: List[float]) -> Dict:
        """
        Evaluate a selected skill set.
        
        Args:
            skills: List of skill IDs to evaluate
        
        Returns:
            Dictionary with evaluation metrics
        """
        if self.coverage_matrix is None:
            self.split_timeline()
            self.compute_dual_coverage()
        
        selected_set = set(skills)
        valid_users = [user for user, user_skills in self.coverage_matrix.items()
                      if selected_set.issubset(user_skills)]
        
        # Compute additional metrics
        skill_user_counts = self.df[self.df['skill_id'].isin(skills)].groupby('skill_id')['user_id'].nunique()
        skill_problem_counts = self.df[self.df['skill_id'].isin(skills)].groupby('skill_id').size()
        
        return {
            'n_skills': len(skills),
            'n_valid_users': len(valid_users),
            'valid_user_ids': valid_users,
            'skills': skills,
            'coverage_rate': len(valid_users) / len(self.coverage_matrix) if self.coverage_matrix else 0,
            'skill_user_counts': skill_user_counts.to_dict(),
            'skill_problem_counts': skill_problem_counts.to_dict()
        }
    
    def compare_strategies(self, K_values: List[int] = [5, 10, 15], 
                          min_users: int = 500) -> pd.DataFrame:
        """
        Compare all three strategies for different K values.
        
        Args:
            K_values: List of K values to test
            min_users: Minimum users threshold
        
        Returns:
            DataFrame with comparison results
        """
        results = []
        
        for K in K_values:
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Evaluating K={K}")
                print(f"{'='*60}")
            
            # Strategy A
            skills_a = self.strategy_frequency(K, min_users)
            eval_a = self.evaluate_skillset(skills_a)
            results.append({
                'K': K,
                'strategy': 'A_Frequency',
                'n_valid_users': eval_a['n_valid_users'],
                'coverage_rate': eval_a['coverage_rate'],
                'skills': eval_a['skills']
            })
            
            # Strategy B
            skills_b, n_users_b = self.strategy_cooccurrence(K, min_users)
            eval_b = self.evaluate_skillset(skills_b)
            results.append({
                'K': K,
                'strategy': 'B_Cooccurrence',
                'n_valid_users': eval_b['n_valid_users'],
                'coverage_rate': eval_b['coverage_rate'],
                'skills': eval_b['skills']
            })
            
            # Strategy C
            skills_c = self.strategy_hybrid(K, min_users)
            eval_c = self.evaluate_skillset(skills_c)
            results.append({
                'K': K,
                'strategy': 'C_Hybrid',
                'n_valid_users': eval_c['n_valid_users'],
                'coverage_rate': eval_c['coverage_rate'],
                'skills': eval_c['skills']
            })
        
        return pd.DataFrame(results)


def main():
    """Example usage"""
    # Load data
    data_path = Path("data/raw/assistments_2009_2010/skill_builder_data.csv")
    df = pd.read_csv(data_path, encoding='latin1')
    
    # Filter to skills with sufficient users
    MIN_USERS = 20
    skill_counts = df.groupby('skill_id')['user_id'].nunique()
    valid_skills = skill_counts[skill_counts >= MIN_USERS].index
    df_filtered = df[df['skill_id'].isin(valid_skills)]
    
    print(f"Filtered to {len(df_filtered)} records with {df_filtered['skill_id'].nunique()} skills")
    
    # Run comparison
    selector = SkillSetSelector(df_filtered)
    comparison_df = selector.compare_strategies(K_values=[5, 10, 15], min_users=MIN_USERS)
    
    # Display results
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    print(comparison_df[['K', 'strategy', 'n_valid_users', 'coverage_rate']])
    
    # Save results
    output_dir = Path("outputs/assistments_2009_2010/skillset_selection")
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(output_dir / "strategy_comparison.csv", index=False)
    print(f"\nResults saved to {output_dir / 'strategy_comparison.csv'}")


if __name__ == "__main__":
    main()
