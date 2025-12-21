"""
Extract data for selected skill sets from ASSISTments 2009-2010 dataset.

This script:
1. Loads saved skill set configurations
2. Extracts data for valid users and selected skills
3. Splits data into first-half and second-half based on temporal order
4. Saves processed data for DINA model analysis
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
import argparse


class DataExtractor:
    """Extract and process data for DINA model analysis."""
    
    def __init__(self, data_path: str, config_path: str, verbose: bool = True):
        """
        Initialize DataExtractor.
        
        Args:
            data_path: Path to raw skill_builder_data.csv
            config_path: Path to configuration JSON file
            verbose: Whether to print progress information
        """
        self.verbose = verbose
        
        # Load raw data
        if self.verbose:
            print(f"Loading data from {data_path}...")
        self.df = pd.read_csv(data_path, encoding='latin1')
        
        # Load configuration
        if self.verbose:
            print(f"Loading configuration from {config_path}...")
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.selected_skills = self.config['selected_skills']
        self.K = self.config['K']
        self.min_users = self.config['min_users_threshold']
        
        if self.verbose:
            print(f"Configuration: K={self.K}, MIN_USERS={self.min_users}")
            print(f"Selected skills: {self.selected_skills}")
    
    def load_valid_users(self, valid_users_path: str) -> List[int]:
        """Load valid user IDs from CSV file."""
        if self.verbose:
            print(f"Loading valid users from {valid_users_path}...")
        
        users_df = pd.read_csv(valid_users_path)
        user_ids = users_df['user_id'].tolist()
        
        if self.verbose:
            print(f"Loaded {len(user_ids)} valid users")
        
        return user_ids
    
    def filter_data(self, user_ids: List[int]) -> pd.DataFrame:
        """Filter data for selected users and skills."""
        if self.verbose:
            print("Filtering data...")
            print(f"  Original: {len(self.df)} records, {self.df['user_id'].nunique()} users")
        
        # Filter by users and skills
        df_filtered = self.df[
            (self.df['user_id'].isin(user_ids)) &
            (self.df['skill_id'].isin(self.selected_skills))
        ].copy()
        
        # Sort by user and order
        df_filtered = df_filtered.sort_values(['user_id', 'order_id']).reset_index(drop=True)
        
        if self.verbose:
            print(f"  Filtered: {len(df_filtered)} records, {df_filtered['user_id'].nunique()} users")
            print(f"  Skills: {df_filtered['skill_id'].nunique()}")
        
        return df_filtered
    
    def split_temporal_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[int]]:
        """
        Split data into first-half and second-half for each user.
        Only keep users who attempted all skills in BOTH periods.
        
        Args:
            df: Filtered dataframe
            
        Returns:
            df_first: First-half data
            df_second: Second-half data
            valid_users: List of user IDs with complete data in both periods
        """
        if self.verbose:
            print("Splitting data into first-half and second-half...")
        
        first_half_indices = []
        second_half_indices = []
        
        for user_id, user_data in df.groupby('user_id'):
            n_problems = len(user_data)
            split_idx = n_problems // 2
            
            indices = user_data.index.tolist()
            first_half_indices.extend(indices[:split_idx])
            second_half_indices.extend(indices[split_idx:])
        
        df_first = df.loc[first_half_indices].copy()
        df_second = df.loc[second_half_indices].copy()
        
        if self.verbose:
            print(f"  First-half: {len(df_first)} records, {df_first['user_id'].nunique()} users")
            print(f"  Second-half: {len(df_second)} records, {df_second['user_id'].nunique()} users")
        
        # Validate: Keep only users who attempted all skills in BOTH periods
        if self.verbose:
            print("\nValidating temporal coverage...")
        
        users_first = set()
        for user_id, user_data in df_first.groupby('user_id'):
            skills_attempted = user_data['skill_id'].unique()
            if len(skills_attempted) == self.K:
                users_first.add(user_id)
        
        users_second = set()
        for user_id, user_data in df_second.groupby('user_id'):
            skills_attempted = user_data['skill_id'].unique()
            if len(skills_attempted) == self.K:
                users_second.add(user_id)
        
        # Keep only users with complete coverage in both periods
        valid_users = sorted(users_first & users_second)
        
        if self.verbose:
            print(f"  Users with all {self.K} skills in first-half: {len(users_first)}")
            print(f"  Users with all {self.K} skills in second-half: {len(users_second)}")
            print(f"  Users with all {self.K} skills in BOTH periods: {len(valid_users)}")
        
        # Filter data to keep only valid users
        df_first = df_first[df_first['user_id'].isin(valid_users)].copy()
        df_second = df_second[df_second['user_id'].isin(valid_users)].copy()
        
        if self.verbose:
            print(f"\n  Final first-half: {len(df_first)} records, {df_first['user_id'].nunique()} users")
            print(f"  Final second-half: {len(df_second)} records, {df_second['user_id'].nunique()} users")
        
        return df_first, df_second, valid_users
    
    def create_response_matrix(self, df: pd.DataFrame, user_ids: List[int]) -> pd.DataFrame:
        """
        Create response matrix for DINA model.
        
        Matrix format:
        - Rows: users
        - Columns: skills
        - Values: response pattern (e.g., "1,0,1" for 3 problems)
        
        Args:
            df: Data for one temporal period
            user_ids: List of user IDs to include
            
        Returns:
            Response matrix as DataFrame
        """
        if self.verbose:
            print("Creating response matrix...")
        
        response_data = []
        
        for user_id in user_ids:
            user_data = df[df['user_id'] == user_id]
            row = {'user_id': user_id}
            
            for skill_id in self.selected_skills:
                skill_data = user_data[user_data['skill_id'] == skill_id]
                
                if len(skill_data) > 0:
                    # Get response pattern
                    responses = skill_data['correct'].astype(int).tolist()
                    row[f'skill_{skill_id}'] = ','.join(map(str, responses))
                else:
                    row[f'skill_{skill_id}'] = ''
            
            response_data.append(row)
        
        response_matrix = pd.DataFrame(response_data)
        
        if self.verbose:
            print(f"  Response matrix shape: {response_matrix.shape}")
        
        return response_matrix
    
    def create_aggregated_matrix(self, df: pd.DataFrame, user_ids: List[int]) -> pd.DataFrame:
        """
        Create aggregated response matrix (proportion correct for each skill).
        
        Args:
            df: Data for one temporal period
            user_ids: List of user IDs to include
            
        Returns:
            Aggregated matrix as DataFrame
        """
        if self.verbose:
            print("Creating aggregated matrix...")
        
        agg_data = []
        
        for user_id in user_ids:
            user_data = df[df['user_id'] == user_id]
            row = {'user_id': user_id}
            
            for skill_id in self.selected_skills:
                skill_data = user_data[user_data['skill_id'] == skill_id]
                
                if len(skill_data) > 0:
                    # Calculate proportion correct
                    prop_correct = skill_data['correct'].mean()
                    row[f'skill_{skill_id}'] = prop_correct
                    row[f'skill_{skill_id}_count'] = len(skill_data)
                else:
                    row[f'skill_{skill_id}'] = np.nan
                    row[f'skill_{skill_id}_count'] = 0
            
            agg_data.append(row)
        
        agg_matrix = pd.DataFrame(agg_data)
        
        if self.verbose:
            print(f"  Aggregated matrix shape: {agg_matrix.shape}")
        
        return agg_matrix
    
    def save_results(self, output_dir: str, user_ids: List[int], 
                     df_filtered: pd.DataFrame, df_first: pd.DataFrame, 
                     df_second: pd.DataFrame):
        """
        Save all processed data.
        
        Args:
            output_dir: Output directory path
            user_ids: List of valid user IDs
            df_filtered: Full filtered data
            df_first: First-half data
            df_second: Second-half data
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if self.verbose:
            print(f"\nSaving results to {output_path}...")
        
        # Save configuration info
        info = {
            'K': self.K,
            'min_users': self.min_users,
            'n_users': len(user_ids),
            'selected_skills': self.selected_skills,
            'skill_names': self.config.get('skill_names', []),
            'n_records_total': len(df_filtered),
            'n_records_first': len(df_first),
            'n_records_second': len(df_second)
        }
        
        with open(output_path / 'data_info.json', 'w') as f:
            json.dump(info, f, indent=2)
        if self.verbose:
            print("  ✓ data_info.json")
        
        # Save raw filtered data
        df_filtered.to_csv(output_path / 'filtered_data.csv', index=False)
        if self.verbose:
            print("  ✓ filtered_data.csv")
        
        df_first.to_csv(output_path / 'first_half_data.csv', index=False)
        if self.verbose:
            print("  ✓ first_half_data.csv")
        
        df_second.to_csv(output_path / 'second_half_data.csv', index=False)
        if self.verbose:
            print("  ✓ second_half_data.csv")
        
        # Create and save response matrices
        response_first = self.create_response_matrix(df_first, user_ids)
        response_first.to_csv(output_path / 'response_matrix_first.csv', index=False)
        if self.verbose:
            print("  ✓ response_matrix_first.csv")
        
        response_second = self.create_response_matrix(df_second, user_ids)
        response_second.to_csv(output_path / 'response_matrix_second.csv', index=False)
        if self.verbose:
            print("  ✓ response_matrix_second.csv")
        
        # Create and save aggregated matrices
        agg_first = self.create_aggregated_matrix(df_first, user_ids)
        agg_first.to_csv(output_path / 'aggregated_matrix_first.csv', index=False)
        if self.verbose:
            print("  ✓ aggregated_matrix_first.csv")
        
        agg_second = self.create_aggregated_matrix(df_second, user_ids)
        agg_second.to_csv(output_path / 'aggregated_matrix_second.csv', index=False)
        if self.verbose:
            print("  ✓ aggregated_matrix_second.csv")
        
        # Save user statistics
        user_stats = []
        for user_id in user_ids:
            total_count = len(df_filtered[df_filtered['user_id'] == user_id])
            first_count = len(df_first[df_first['user_id'] == user_id])
            second_count = len(df_second[df_second['user_id'] == user_id])
            
            user_stats.append({
                'user_id': user_id,
                'total_problems': total_count,
                'first_half_problems': first_count,
                'second_half_problems': second_count
            })
        
        pd.DataFrame(user_stats).to_csv(output_path / 'user_statistics.csv', index=False)
        if self.verbose:
            print("  ✓ user_statistics.csv")
        
        if self.verbose:
            print(f"\n✓ All data saved to {output_path}")
    
    def run(self, valid_users_path: str, output_dir: str):
        """
        Execute full data extraction pipeline.
        
        Args:
            valid_users_path: Path to valid users CSV
            output_dir: Output directory path
        """
        print("="*80)
        print("DATA EXTRACTION PIPELINE")
        print("="*80)
        
        # Load valid users
        initial_user_ids = self.load_valid_users(valid_users_path)
        
        # Filter data
        df_filtered = self.filter_data(initial_user_ids)
        
        # Split temporal data and validate coverage
        df_first, df_second, valid_user_ids = self.split_temporal_data(df_filtered)
        
        # Re-filter full data to keep only validated users
        df_filtered = df_filtered[df_filtered['user_id'].isin(valid_user_ids)].copy()
        
        # Save results with validated users only
        self.save_results(output_dir, valid_user_ids, df_filtered, df_first, df_second)
        
        print("="*80)
        print("EXTRACTION COMPLETE")
        print("="*80)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Extract data for selected skill sets from ASSISTments dataset'
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
        '--data_path',
        type=str,
        default='data/raw/assistments_2009_2010/skill_builder_data.csv',
        help='Path to raw data CSV'
    )
    parser.add_argument(
        '--config_dir',
        type=str,
        default='outputs/assistments_2009_2010/skillset_selection',
        help='Directory containing configuration files'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory (default: data/processed/assistments_2009_2010/min{MIN_USERS}_k{K})'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Print progress information'
    )
    
    args = parser.parse_args()
    
    # Construct paths
    config_path = Path(args.config_dir) / f'config_min{args.min_users}_k{args.K}.json'
    valid_users_path = Path(args.config_dir) / f'valid_users_min{args.min_users}_k{args.K}.csv'
    
    if args.output_dir is None:
        output_dir = f'data/processed/assistments_2009_2010/min{args.min_users}_k{args.K}'
    else:
        output_dir = args.output_dir
    
    # Check if files exist
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    if not valid_users_path.exists():
        raise FileNotFoundError(f"Valid users file not found: {valid_users_path}")
    
    # Run extraction
    extractor = DataExtractor(args.data_path, str(config_path), verbose=args.verbose)
    extractor.run(str(valid_users_path), output_dir)
    
    print(f"\n✓ Data extraction completed successfully!")
    print(f"  Output directory: {output_dir}")


if __name__ == '__main__':
    main()

"""
data_info.json - 設定情報
filtered_data.csv - フィルタ済み全データ
first_half_data.csv / second_half_data.csv - 前半・後半データ
response_matrix_first.csv / response_matrix_second.csv - 応答パターン行列
aggregated_matrix_first.csv / aggregated_matrix_second.csv - 正答率集計
user_statistics.csv - ユーザー統計
"""