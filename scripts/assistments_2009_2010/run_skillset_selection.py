"""
Run skill set selection with improved temporal coverage guarantee.

This script runs Strategy B (Co-occurrence) with temporal coverage requirement
to maximize the number of users who have all selected skills in BOTH
first and second halves of their problem sequences.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import argparse
import sys

# Add scripts directory to path
script_dir = Path(__file__).parent.parent
sys.path.insert(0, str(script_dir))

from assistments_2009_2010.select_skillset import SkillSetSelector


def run_selection(K: int, min_users: int, output_dir: str, data_path: str):
    """
    Run skill set selection with temporal coverage requirement.
    
    Args:
        K: Number of skills to select
        min_users: Minimum users threshold
        output_dir: Output directory
        data_path: Path to raw data
    """
    print("="*80)
    print(f"SKILL SET SELECTION (K={K}, MIN_USERS={min_users})")
    print("="*80)
    
    # Load data
    print(f"\nLoading data from {data_path}...")
    df = pd.read_csv(data_path, encoding='latin1')
    print(f"  Total records: {len(df)}")
    print(f"  Total users: {df['user_id'].nunique()}")
    print(f"  Total skills: {df['skill_id'].nunique()}")
    
    # Initialize selector
    selector = SkillSetSelector(df, verbose=True)
    
    # Run Strategy B with temporal coverage requirement
    print(f"\nRunning Strategy B (Co-occurrence with temporal coverage)...")
    selected_skills, n_valid_users = selector.strategy_cooccurrence(
        K=K, 
        min_users=min_users,
        require_temporal_coverage=True
    )
    
    # Evaluate
    evaluation = selector.evaluate_skillset(selected_skills)
    
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")
    print(f"Selected {len(selected_skills)} skills")
    print(f"Valid users (both halves): {evaluation['n_valid_users']}")
    print(f"Coverage rate: {evaluation['coverage_rate']:.1%}")
    
    # Get skill names
    skill_names = {}
    for skill_id in selected_skills:
        skill_data = df[df['skill_id'] == skill_id]
        if 'skill_name' in df.columns:
            name = skill_data['skill_name'].iloc[0] if len(skill_data) > 0 else f"Skill {skill_id}"
        else:
            name = f"Skill {skill_id}"
        skill_names[skill_id] = name
    
    print(f"\nSelected skills:")
    for i, skill_id in enumerate(selected_skills, 1):
        user_count = evaluation['skill_user_counts'].get(skill_id, 0)
        problem_count = evaluation['skill_problem_counts'].get(skill_id, 0)
        print(f"  {i:2d}. {skill_names[skill_id]}")
        print(f"      ID: {skill_id}, Users: {user_count}, Problems: {problem_count}")
    
    # Save configuration
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    config = {
        'description': 'Temporal coverage optimized',
        'min_users_threshold': min_users,
        'K': K,
        'strategy': 'B_Cooccurrence_Temporal',
        'n_valid_users': evaluation['n_valid_users'],
        'selected_skills': selected_skills,
        'skill_names': [skill_names[s] for s in selected_skills],
        'skill_user_counts': {str(k): int(v) for k, v in evaluation['skill_user_counts'].items()},
        'skill_problem_counts': {str(k): int(v) for k, v in evaluation['skill_problem_counts'].items()},
        'coverage_rate': evaluation['coverage_rate']
    }
    
    config_file = output_path / f'config_min{min_users}_k{K}.json'
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\n✓ Configuration saved to {config_file}")
    
    # Save valid user IDs
    valid_users_df = pd.DataFrame({'user_id': evaluation['valid_user_ids']})
    users_file = output_path / f'valid_users_min{min_users}_k{K}.csv'
    valid_users_df.to_csv(users_file, index=False)
    print(f"✓ Valid users saved to {users_file}")
    
    print(f"\n{'='*80}")
    print("SELECTION COMPLETE")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description='Run skill set selection with temporal coverage optimization'
    )
    parser.add_argument(
        '--K',
        type=int,
        default=10,
        help='Number of skills to select (default: 10)'
    )
    parser.add_argument(
        '--min_users',
        type=int,
        default=150,
        help='Minimum users threshold (default: 150)'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default='data/raw/assistments_2009_2010/skill_builder_data.csv',
        help='Path to raw data CSV'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs/assistments_2009_2010/skillset_selection',
        help='Output directory'
    )
    
    args = parser.parse_args()
    
    run_selection(
        K=args.K,
        min_users=args.min_users,
        output_dir=args.output_dir,
        data_path=args.data_path
    )


if __name__ == '__main__':
    main()
