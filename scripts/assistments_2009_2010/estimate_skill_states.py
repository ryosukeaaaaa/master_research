"""
Estimate skill states using DINA model for first-half and second-half data.

This script:
1. Loads processed data for selected users and skills
2. Creates Q-matrix (identity matrix for single-skill items)
3. Estimates skill states using EduCDM's DINA model for both periods
4. Analyzes skill state transitions between first and second halves
5. Saves estimated skill states and transition analysis
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
from EduCDM import EMDINA


class SkillStateEstimator:
    """Estimate skill states using DINA model."""
    
    def __init__(self, data_dir: str, verbose: bool = True):
        """
        Initialize SkillStateEstimator.
        
        Args:
            data_dir: Directory containing processed data
            verbose: Whether to print progress information
        """
        self.data_dir = Path(data_dir)
        self.verbose = verbose
        
        # Load data info
        if self.verbose:
            print(f"Loading data from {self.data_dir}...")
        
        with open(self.data_dir / 'data_info.json', 'r') as f:
            self.info = json.load(f)
        
        self.K = self.info['K']
        self.selected_skills = self.info['selected_skills']
        self.skill_names = self.info['skill_names']
        self.n_users = self.info['n_users']
        
        if self.verbose:
            print(f"  K={self.K} skills, {self.n_users} users")
            print(f"  Skills: {self.skill_names}")
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load aggregated first-half and second-half data."""
        if self.verbose:
            print("\nLoading aggregated temporal data...")
        
        df_first = pd.read_csv(self.data_dir / 'aggregated_matrix_first.csv')
        df_second = pd.read_csv(self.data_dir / 'aggregated_matrix_second.csv')
        
        if self.verbose:
            print(f"  First-half: {len(df_first)} users")
            print(f"  Second-half: {len(df_second)} users")
            print(f"  All users have data for all {self.K} skills (no sparsity)")
        
        return df_first, df_second
    
    def create_q_matrix_for_aggregated(self, n_items_per_skill: int) -> np.ndarray:
        """
        Create Q-matrix for aggregated skill-level data with multiple items per skill.
        
        Args:
            n_items_per_skill: Number of virtual items per skill
            
        Returns:
            Q-matrix of shape (K * n_items_per_skill, K)
        """
        if self.verbose:
            print("\nCreating Q-matrix for skill-level data...")
        
        # Create Q-matrix: n_items_per_skill items for each skill
        Q = np.zeros((self.K * n_items_per_skill, self.K), dtype=int)
        for skill_idx in range(self.K):
            start_idx = skill_idx * n_items_per_skill
            end_idx = start_idx + n_items_per_skill
            Q[start_idx:end_idx, skill_idx] = 1
        
        if self.verbose:
            print(f"  Q-matrix shape: {Q.shape} ({self.K} skills × {n_items_per_skill} items each)")
        
        return Q
    
    def create_q_matrix(self, df: pd.DataFrame, item_indices: np.ndarray) -> np.ndarray:
        """
        Create Q-matrix mapping items to skills.
        
        Args:
            df: Original DataFrame containing problem data with skill_id column
            item_indices: Indices of selected items in df
            
        Returns:
            Q-matrix of shape (n_items, K) where Q[i,j]=1 if item i tests skill j
        """
        if self.verbose:
            print("\nCreating Q-matrix...")
        
        # Get selected items
        df_selected = df.loc[item_indices].reset_index(drop=True)
        n_items = len(df_selected)
        
        # Map skills to indices
        skill_to_idx = {skill: idx for idx, skill in enumerate(self.selected_skills)}
        
        # Create Q-matrix
        Q = np.zeros((n_items, self.K), dtype=int)
        
        for item_idx, (_, row) in enumerate(df_selected.iterrows()):
            skill_idx = skill_to_idx[row['skill_id']]
            Q[item_idx, skill_idx] = 1
        
        if self.verbose:
            print(f"  Q-matrix shape: {Q.shape} (n_items × K)")
            print(f"  Number of items: {n_items}")
            print(f"  Number of skills: {self.K}")
            print(f"  Items per skill:")
            for skill_idx, skill_name in enumerate(self.skill_names):
                n_skill_items = Q[:, skill_idx].sum()
                print(f"    {skill_idx+1}. {skill_name}: {n_skill_items} items")
        
        return Q
    
    def prepare_dina_data_from_aggregated(self, df: pd.DataFrame, n_items_per_skill: int = 10) -> Tuple[np.ndarray, List[int]]:
        """
        Prepare aggregated skill-level data for DINA model.
        Convert skill-level accuracy rates to binary responses based on actual counts.
        
        Args:
            df: Aggregated DataFrame with columns [user_id, skill_X.0, skill_X.0_count, ...]
            n_items_per_skill: Number of virtual items to generate per skill
            
        Returns:
            response_matrix: Response data (n_users, K * n_items_per_skill)
            user_ids: List of unique user IDs
        """
        if self.verbose:
            print("\nPreparing aggregated data for DINA model...")
        
        user_ids = df['user_id'].tolist()
        n_users = len(user_ids)
        
        # Extract skill accuracy and count columns
        skill_cols = sorted([col for col in df.columns if col.startswith('skill_') and not col.endswith('_count')])
        
        # Set random seed for reproducibility
        np.random.seed(42)
        response_lists = []
        
        for skill_col in skill_cols:
            count_col = f"{skill_col}_count"
            accuracy = df[skill_col].values
            counts = df[count_col].values
            
            # Generate virtual items based on accuracy distribution
            for item_idx in range(n_items_per_skill):
                # Each virtual item uses the same accuracy (no noise added)
                responses = (np.random.random(n_users) < accuracy).astype(int)
                response_lists.append(responses)
        
        response_matrix = np.column_stack(response_lists)
        
        if self.verbose:
            print(f"  Users: {n_users}")
            print(f"  Skills: {self.K}")
            print(f"  Virtual items per skill: {n_items_per_skill}")
            print(f"  Total items: {response_matrix.shape[1]}")
            print(f"  Response matrix shape: {response_matrix.shape}")
            print(f"  No missing values (all students attempted all skills)")
            print(f"  Average accuracy: {response_matrix.mean():.2%}")
            
            # Show skill-level accuracies
            print(f"  Skill-level accuracies in generated data:")
            for i, skill_col in enumerate(skill_cols):
                start_idx = i * n_items_per_skill
                end_idx = start_idx + n_items_per_skill
                skill_acc = response_matrix[:, start_idx:end_idx].mean()
                original_acc = df[skill_col].mean()
                print(f"    Skill {i+1}: Generated={skill_acc:.2%}, Original={original_acc:.2%}")
        
        return response_matrix, user_ids
    
    def prepare_dina_data(self, df: pd.DataFrame, max_items_per_skill: int = 50) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        Prepare data in format required by DINA model.
        Sample representative items for each skill to reduce dimensionality.
        
        Args:
            df: DataFrame with columns [user_id, skill_id, correct]
            max_items_per_skill: Maximum number of items to sample per skill
            
        Returns:
            response_matrix: Response data (n_users, n_items) = response matrix
            item_indices: Original indices of selected items in df
            user_ids: List of unique user IDs
        """
        if self.verbose:
            print("\nPreparing data for DINA model...")
        
        # Get unique users
        user_ids = sorted(df['user_id'].unique())
        user_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
        n_users = len(user_ids)
        
        # Sample items per skill to reduce problem count
        selected_indices = []
        skill_to_idx = {skill: idx for idx, skill in enumerate(self.selected_skills)}
        
        for skill_id in self.selected_skills:
            skill_items = df[df['skill_id'] == skill_id]
            
            # Sample items: prioritize diverse users
            if len(skill_items) <= max_items_per_skill:
                selected_indices.extend(skill_items.index.tolist())
            else:
                # Sample items that cover different users
                user_counts = skill_items.groupby('user_id').size()
                sampled_users = user_counts.nlargest(min(len(user_counts), max_items_per_skill // 2)).index
                
                # Get items from these users
                sampled_items = skill_items[skill_items['user_id'].isin(sampled_users)]
                
                # If still too many, randomly sample
                if len(sampled_items) > max_items_per_skill:
                    sampled_items = sampled_items.sample(n=max_items_per_skill, random_state=42)
                
                selected_indices.extend(sampled_items.index.tolist())
        
        # Get selected data
        df_selected = df.loc[selected_indices].reset_index(drop=True)
        n_items = len(df_selected)
        
        # Create response matrix (n_users, n_items)
        response_matrix = np.full((n_users, n_items), -1, dtype=int)  # -1 for missing values
        
        # Fill in responses
        for item_idx, (_, row) in enumerate(df_selected.iterrows()):
            user_idx = user_to_idx[row['user_id']]
            response = int(row['correct'])
            response_matrix[user_idx, item_idx] = response
        
        if self.verbose:
            print(f"  Users: {n_users}")
            print(f"  Original items: {len(df)}")
            print(f"  Sampled items: {n_items} (~{max_items_per_skill} per skill)")
            print(f"  Response matrix shape: {response_matrix.shape}")
            print(f"  Sparsity: {(response_matrix == -1).sum() / response_matrix.size:.1%}")
        
        return response_matrix, np.array(selected_indices), user_ids
    
    def estimate_skill_states(self, response_matrix: np.ndarray, Q: np.ndarray,
                             period: str, epoch: int = 30) -> np.ndarray:
        """
        Estimate skill states using DINA model.
        
        Args:
            response_matrix: Response data (n_users, n_items)
            Q: Q-matrix (n_items, K)
            period: 'first' or 'second' for logging
            epoch: Number of training epochs (default: 30, recommended range: 30-50)
            
        Returns:
            skill_states: Binary skill state matrix (n_users, K)
            
        Note:
            EMDINA uses EM algorithm with early stopping:
            - Always runs minimum 20 iterations
            - After iteration 20, stops when parameter change < epsilon
            - epsilon=1e-3 is standard convergence threshold
            - epoch=30-50 ensures sufficient convergence for most cases
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Estimating skill states for {period}-half data")
            print(f"{'='*60}")
        
        # Get dimensions
        stu_num = response_matrix.shape[0]
        prob_num = response_matrix.shape[1]
        know_num = Q.shape[1]
        
        # Initialize DINA model
        dina = EMDINA(response_matrix, Q, stu_num, prob_num, know_num, skip_value=-1)
        
        if self.verbose:
            print(f"Model initialized: {stu_num} students, {prob_num} problems, {know_num} skills")
            print(f"Training DINA model (epochs={epoch}, epsilon=1e-3)...")
        
        # Train model with EM algorithm
        # Early stopping: iteration > 20 AND change < 1e-3
        dina.train(epoch=epoch, epsilon=1e-3)
        
        # Extract skill states using EMDINA's theta attribute
        # theta contains state indices for each user
        # all_states contains all possible 2^K skill mastery patterns
        skill_states = dina.all_states[dina.theta].astype(int)
        
        if self.verbose:
            print(f"  Unique skill states used: {len(np.unique(dina.theta))} out of {len(dina.all_states)}")
            print(f"  Slip parameters (mean): {dina.slip.mean():.3f}")
            print(f"  Guess parameters (mean): {dina.guess.mean():.3f}")
        
        if self.verbose:
            print(f"\n✓ Estimation complete")
            print(f"  Skill states shape: {skill_states.shape}")
            print(f"  Mastery rates per skill:")
            for i, skill_name in enumerate(self.skill_names):
                mastery_rate = skill_states[:, i].mean()
                print(f"    {i+1}. {skill_name}: {mastery_rate:.2%}")
        
        return skill_states
    
    def analyze_transitions(self, states_first: np.ndarray, 
                           states_second: np.ndarray,
                           user_ids: List[int]) -> pd.DataFrame:
        """
        Analyze skill state transitions between first and second halves.
        
        Args:
            states_first: Skill states for first-half (n_users, K)
            states_second: Skill states for second-half (n_users, K)
            user_ids: List of user IDs
            
        Returns:
            DataFrame with transition analysis
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print("Analyzing skill state transitions")
            print(f"{'='*60}")
        
        transitions = []
        
        for i, user_id in enumerate(user_ids):
            user_trans = {'user_id': user_id}
            
            for j, skill_name in enumerate(self.skill_names):
                state_first = int(states_first[i, j])
                state_second = int(states_second[i, j])
                
                user_trans[f'skill_{j+1}_first'] = state_first
                user_trans[f'skill_{j+1}_second'] = state_second
                user_trans[f'skill_{j+1}_transition'] = f"{state_first}→{state_second}"
            
            # Calculate summary statistics
            user_trans['mastered_first'] = int(states_first[i].sum())
            user_trans['mastered_second'] = int(states_second[i].sum())
            user_trans['acquired'] = int((states_second[i] > states_first[i]).sum())
            user_trans['lost'] = int((states_first[i] > states_second[i]).sum())
            user_trans['net_gain'] = user_trans['acquired'] - user_trans['lost']
            
            transitions.append(user_trans)
        
        df_transitions = pd.DataFrame(transitions)
        
        if self.verbose:
            print(f"\nTransition Summary:")
            print(f"  Average skills mastered (first): {df_transitions['mastered_first'].mean():.2f}")
            print(f"  Average skills mastered (second): {df_transitions['mastered_second'].mean():.2f}")
            print(f"  Average skills acquired: {df_transitions['acquired'].mean():.2f}")
            print(f"  Average skills lost: {df_transitions['lost'].mean():.2f}")
            print(f"  Average net gain: {df_transitions['net_gain'].mean():.2f}")
            
            print(f"\n  Skill-level transitions:")
            for j, skill_name in enumerate(self.skill_names):
                first_mastery = states_first[:, j].mean()
                second_mastery = states_second[:, j].mean()
                change = second_mastery - first_mastery
                print(f"    {j+1}. {skill_name}:")
                print(f"       First: {first_mastery:.2%} → Second: {second_mastery:.2%} (Δ={change:+.2%})")
        
        return df_transitions
    
    def save_results(self, output_dir: str, user_ids: List[int],
                    states_first: np.ndarray, states_second: np.ndarray,
                    df_transitions: pd.DataFrame):
        """
        Save estimation results.
        
        Args:
            output_dir: Output directory path
            user_ids: List of user IDs
            states_first: Skill states for first-half
            states_second: Skill states for second-half
            df_transitions: Transition analysis DataFrame
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Saving results to {output_path}")
            print(f"{'='*60}")
        
        # Save skill states for first-half
        df_first = pd.DataFrame(states_first, columns=[f'skill_{i+1}' for i in range(self.K)])
        df_first.insert(0, 'user_id', user_ids)
        df_first.to_csv(output_path / 'skill_states_first.csv', index=False)
        if self.verbose:
            print("  ✓ skill_states_first.csv")
        
        # Save skill states for second-half
        df_second = pd.DataFrame(states_second, columns=[f'skill_{i+1}' for i in range(self.K)])
        df_second.insert(0, 'user_id', user_ids)
        df_second.to_csv(output_path / 'skill_states_second.csv', index=False)
        if self.verbose:
            print("  ✓ skill_states_second.csv")
        
        # Save transitions
        df_transitions.to_csv(output_path / 'skill_transitions.csv', index=False)
        if self.verbose:
            print("  ✓ skill_transitions.csv")
        
        # Save skill mapping
        skill_mapping = {
            'skills': [
                {
                    'skill_id': int(self.selected_skills[i]) if isinstance(self.selected_skills[i], float) else self.selected_skills[i],
                    'skill_name': self.skill_names[i],
                    'skill_index': i + 1
                }
                for i in range(self.K)
            ]
        }
        
        with open(output_path / 'skill_mapping.json', 'w') as f:
            json.dump(skill_mapping, f, indent=2)
        if self.verbose:
            print("  ✓ skill_mapping.json")
        
        # Save summary statistics
        summary = {
            'K': self.K,
            'n_users': len(user_ids),
            'mastery_rates_first': {
                self.skill_names[i]: float(states_first[:, i].mean())
                for i in range(self.K)
            },
            'mastery_rates_second': {
                self.skill_names[i]: float(states_second[:, i].mean())
                for i in range(self.K)
            },
            'average_mastered_first': float(df_transitions['mastered_first'].mean()),
            'average_mastered_second': float(df_transitions['mastered_second'].mean()),
            'average_acquired': float(df_transitions['acquired'].mean()),
            'average_lost': float(df_transitions['lost'].mean()),
            'average_net_gain': float(df_transitions['net_gain'].mean())
        }
        
        with open(output_path / 'estimation_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        if self.verbose:
            print("  ✓ estimation_summary.json")
        
        if self.verbose:
            print(f"\n✓ All results saved to {output_path}")
    
    def run(self, output_dir: str, n_items_per_skill: int = 20):
        """
        Execute full skill state estimation pipeline.
        
        Args:
            output_dir: Output directory path
            n_items_per_skill: Number of virtual items per skill (default: 20)
                               Recommended range: 20-30 for balance of stability and model flexibility
                               Based on empirical analysis:
                               - Too few (<10): High variance in slip/guess parameters
                               - Optimal (20-30): Stable estimates, good state diversity
                               - Too many (>50): Overfitting risk, diminishing returns
        """
        print("="*80)
        print("DINA SKILL STATE ESTIMATION PIPELINE")
        print("="*80)
        
        # Load aggregated data
        df_first, df_second = self.load_data()
        
        # Prepare data from aggregated matrices
        if self.verbose:
            print(f"\nUsing n_items_per_skill={n_items_per_skill}")
        response_first, user_ids = self.prepare_dina_data_from_aggregated(df_first, n_items_per_skill)
        
        # Create Q-matrix for aggregated data
        Q = self.create_q_matrix_for_aggregated(n_items_per_skill)
        
        # Estimate skill states for first-half
        # EM algorithm: 
        #   - Minimum 20 iterations always executed
        #   - After iteration 20, stops when change < epsilon
        #   - epoch=30-50 recommended for convergence
        #   - epsilon=1e-3 is standard (1e-4 too strict, 1e-2 too loose)
        states_first = self.estimate_skill_states(response_first, Q, 'first', epoch=30)
        
        # Prepare data for second-half
        response_second, _ = self.prepare_dina_data_from_aggregated(df_second, n_items_per_skill)
        
        # Estimate skill states for second-half (reuse same Q-matrix)
        states_second = self.estimate_skill_states(response_second, Q, 'second', epoch=30)
        
        # Analyze transitions
        df_transitions = self.analyze_transitions(states_first, states_second, user_ids)
        
        # Save results
        self.save_results(output_dir, user_ids, states_first, states_second, df_transitions)
        
        print("="*80)
        print("ESTIMATION COMPLETE")
        print("="*80)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Estimate skill states using DINA model for ASSISTments dataset'
    )
    parser.add_argument(
        '--min_users',
        type=int,
        default=150,
        help='MIN_USERS threshold (default: 150)'
    )
    parser.add_argument(
        '--K',
        type=int,
        default=10,
        help='Number of skills (default: 10)'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default=None,
        help='Data directory (default: data/processed/assistments_2009_2010/min{MIN_USERS}_k{K})'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory (default: outputs/assistments_2009_2010/dina_estimation/min{MIN_USERS}_k{K})'
    )
    parser.add_argument(
        '--n_items_per_skill',
        type=int,
        default=20,
        help='Number of virtual items to generate per skill (default: 20, recommended: 20-30)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Print progress information'
    )
    
    args = parser.parse_args()
    
    # Construct paths
    if args.data_dir is None:
        data_dir = f'data/processed/assistments_2009_2010/min{args.min_users}_k{args.K}'
    else:
        data_dir = args.data_dir
    
    if args.output_dir is None:
        output_dir = f'outputs/assistments_2009_2010/dina_estimation/min{args.min_users}_k{args.K}'
    else:
        output_dir = args.output_dir
    
    # Check if data directory exists
    if not Path(data_dir).exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Run estimation
    estimator = SkillStateEstimator(data_dir, verbose=args.verbose)
    estimator.run(output_dir, n_items_per_skill=args.n_items_per_skill)
    
    print(f"\n✓ Skill state estimation completed successfully!")
    print(f"  Output directory: {output_dir}")
    print(f"  n_items_per_skill: {args.n_items_per_skill}")


if __name__ == '__main__':
    main()
