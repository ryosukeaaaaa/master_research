"""
指定したスキルセットに合致するユーザを抽出するスクリプト。
Create skill set configuration manually by specifying skill names or IDs.

This script allows you to manually select skills by their names (e.g., "Box and Whisker")
or IDs, validates the selection, and creates configuration files for downstream processing.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import argparse
from typing import List, Set, Dict, Tuple, Union


def load_skill_mapping(data_path: str, verbose: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """
    Load skill mapping from raw data.
    
    Args:
        data_path: Path to skill_builder_data.csv
        verbose: Whether to print progress
        
    Returns:
        df: Full DataFrame
        skill_mapping: Dict mapping skill_name to skill_id
    """
    if verbose:
        print(f"Loading data from {data_path}...")
    
    df = pd.read_csv(data_path, encoding='latin1')
    
    # Create skill name to ID mapping
    skill_info = df[['skill_id', 'skill_name']].dropna().drop_duplicates()
    skill_mapping = dict(zip(skill_info['skill_name'], skill_info['skill_id']))
    
    if verbose:
        print(f"  Total records: {len(df)}")
        print(f"  Total users: {df['user_id'].nunique()}")
        print(f"  Total skills: {df['skill_id'].nunique()}")
        print(f"  Unique skill names: {len(skill_mapping)}")
    
    return df, skill_mapping


def resolve_skill_identifiers(
    identifiers: List[Union[str, float]], 
    skill_mapping: Dict[str, float],
    df: pd.DataFrame
) -> Tuple[List[float], Dict[float, str]]:
    """
    Resolve skill names or IDs to skill IDs.
    
    Args:
        identifiers: List of skill names (str) or IDs (float)
        skill_mapping: Mapping from skill_name to skill_id
        df: DataFrame with skill information
        
    Returns:
        skill_ids: List of resolved skill IDs
        skill_names: Dict mapping skill_id to skill_name
    """
    skill_ids = []
    skill_names = {}
    
    for identifier in identifiers:
        # Try to convert to float (if it's a numeric ID)
        try:
            skill_id = float(identifier)
            # Check if this ID exists
            if skill_id in df['skill_id'].values:
                skill_ids.append(skill_id)
                # Get name for this ID
                name_match = df[df['skill_id'] == skill_id]['skill_name'].dropna()
                if len(name_match) > 0:
                    skill_names[skill_id] = name_match.iloc[0]
                else:
                    skill_names[skill_id] = f"Skill {skill_id}"
            else:
                raise ValueError(f"Skill ID {skill_id} not found in dataset")
        except ValueError:
            # It's a string, treat as skill name
            if identifier in skill_mapping:
                skill_id = skill_mapping[identifier]
                skill_ids.append(skill_id)
                skill_names[skill_id] = identifier
            else:
                # Try case-insensitive match
                matching = [name for name in skill_mapping.keys() 
                          if name.lower() == identifier.lower()]
                if matching:
                    skill_name = matching[0]
                    skill_id = skill_mapping[skill_name]
                    skill_ids.append(skill_id)
                    skill_names[skill_id] = skill_name
                else:
                    raise ValueError(
                        f"Skill name '{identifier}' not found in dataset. "
                        f"Available skills: {list(skill_mapping.keys())[:10]}..."
                    )
    
    return skill_ids, skill_names


def validate_temporal_coverage(
    df: pd.DataFrame, 
    skill_ids: List[float], 
    verbose: bool = True
) -> Tuple[List[int], Dict, Dict]:
    """
    Validate that users have coverage of all skills in both first and second halves.
    
    Args:
        df: Raw data DataFrame
        skill_ids: List of skill IDs to validate
        verbose: Whether to print progress
        
    Returns:
        (valid_user_ids, skill_user_counts, skill_problem_counts)
    """
    if verbose:
        print("\n" + "="*80)
        print("TEMPORAL COVERAGE VALIDATION")
        print("="*80)
    
    # Filter to selected skills
    df_filtered = df[df['skill_id'].isin(skill_ids)].copy()
    df_filtered = df_filtered.sort_values(['user_id', 'order_id']).reset_index(drop=True)
    
    if verbose:
        print(f"Filtered data: {len(df_filtered)} records")
        print(f"Users in filtered data: {df_filtered['user_id'].nunique()}")
    
    # Split each user's timeline into first and second halves
    def get_user_half_skills(user_data):
        n = len(user_data)
        mid = n // 2
        first_skills = set(user_data.iloc[:mid]['skill_id'].dropna())
        second_skills = set(user_data.iloc[mid:]['skill_id'].dropna())
        return first_skills, second_skills
    
    valid_users = []
    skill_set = set(skill_ids)
    
    # Track statistics
    users_with_first = []
    users_with_second = []
    
    if verbose:
        print(f"\nChecking temporal coverage (all {len(skill_ids)} skills in both halves)...")
    
    for user_id, user_data in df_filtered.groupby('user_id'):
        first_skills, second_skills = get_user_half_skills(user_data)
        
        has_first = skill_set.issubset(first_skills)
        has_second = skill_set.issubset(second_skills)
        
        if has_first:
            users_with_first.append(user_id)
        if has_second:
            users_with_second.append(user_id)
        
        # User must have all skills in both halves
        if has_first and has_second:
            valid_users.append(user_id)
    
    if verbose:
        print(f"  Users with all skills in first half: {len(users_with_first)}")
        print(f"  Users with all skills in second half: {len(users_with_second)}")
        print(f"  Users with all skills in BOTH halves: {len(valid_users)}")
    
    # Compute skill statistics (among valid users only)
    df_valid = df_filtered[df_filtered['user_id'].isin(valid_users)]
    skill_user_counts = df_valid.groupby('skill_id')['user_id'].nunique().to_dict()
    skill_problem_counts = df_valid.groupby('skill_id').size().to_dict()
    
    return valid_users, skill_user_counts, skill_problem_counts


def display_verification_summary(
    skill_ids: List[float],
    skill_names: Dict[float, str],
    valid_users: List[int],
    skill_user_counts: Dict,
    skill_problem_counts: Dict,
    total_users: int
):
    """Display comprehensive verification summary."""
    print(f"\n{'='*80}")
    print("DATA VERIFICATION SUMMARY")
    print(f"{'='*80}")
    
    print(f"\n✓ Selected {len(skill_ids)} skills:")
    for i, skill_id in enumerate(skill_ids, 1):
        name = skill_names.get(skill_id, f"Skill {skill_id}")
        user_count = skill_user_counts.get(skill_id, 0)
        problem_count = skill_problem_counts.get(skill_id, 0)
        print(f"  {i:2d}. {name}")
        print(f"      ID: {skill_id}, Users: {user_count}, Problems: {problem_count}")
    
    print(f"\n✓ Valid users: {len(valid_users)}")
    coverage_rate = len(valid_users) / total_users if total_users > 0 else 0
    print(f"  Coverage rate: {coverage_rate:.2%} of all users")
    
    if len(valid_users) > 0:
        print(f"\n✓ Data quality: PASSED")
        print(f"  All {len(skill_ids)} skills are covered by {len(valid_users)} users")
        print(f"  in both first and second halves of their problem sequences.")
    else:
        print(f"\n✗ Data quality: FAILED")
        print(f"  No users have all {len(skill_ids)} skills in both halves.")
        print(f"  Please select different skills.")


def create_manual_configuration(
    data_path: str,
    skill_identifiers: List[Union[str, float]],
    output_dir: str,
    config_name: str = None,
    description: str = "Manually selected skills",
    show_available_skills: bool = False
):
    """
    Create configuration files for manually selected skills.
    
    Args:
        data_path: Path to raw skill_builder_data.csv
        skill_identifiers: List of skill names or IDs
        output_dir: Output directory for configuration files
        config_name: Custom config filename (without extension)
        description: Description for the configuration
        show_available_skills: Whether to display all available skills
    """
    print("="*80)
    print(f"MANUAL SKILL SET CONFIGURATION")
    print("="*80)
    
    # Load data and skill mapping
    df, skill_mapping = load_skill_mapping(data_path, verbose=True)
    
    # Show available skills if requested
    if show_available_skills:
        print(f"\n{'='*80}")
        print(f"AVAILABLE SKILLS (Total: {len(skill_mapping)})")
        print(f"{'='*80}")
        for i, (name, skill_id) in enumerate(sorted(skill_mapping.items()), 1):
            user_count = df[df['skill_id'] == skill_id]['user_id'].nunique()
            problem_count = len(df[df['skill_id'] == skill_id])
            print(f"{i:3d}. {name}")
            print(f"     ID: {skill_id}, Users: {user_count}, Problems: {problem_count}")
        print(f"{'='*80}\n")
        return
    
    # Resolve skill identifiers
    print(f"\nResolving skill identifiers...")
    skill_ids, skill_names = resolve_skill_identifiers(
        skill_identifiers, skill_mapping, df
    )
    
    K = len(skill_ids)
    print(f"✓ Resolved {K} skills")
    
    # Validate temporal coverage
    valid_users, skill_user_counts, skill_problem_counts = validate_temporal_coverage(
        df, skill_ids, verbose=True
    )
    
    # Display verification summary
    display_verification_summary(
        skill_ids, skill_names, valid_users, 
        skill_user_counts, skill_problem_counts,
        df['user_id'].nunique()
    )
    
    if len(valid_users) == 0:
        raise ValueError(
            "\nNo valid users found. Cannot create configuration. "
            "Please select different skills that have better temporal coverage."
        )
    
    # Sort skill_ids for consistent naming
    sorted_skill_ids = sorted(skill_ids)
    
    # Create skill ID string for directory/file naming (e.g., "s70_s77_s280")
    skill_id_str = '_'.join([f's{int(sid)}' for sid in sorted_skill_ids])
    
    # Create configuration (keep original order in config)
    config = {
        'description': description,
        'min_users_threshold': None,  # N/A for manual selection
        'K': K,
        'strategy': 'Manual',
        'n_valid_users': len(valid_users),
        'selected_skills': skill_ids,  # Keep original order
        'skill_names': [skill_names[s] for s in skill_ids],
        'skill_user_counts': {str(int(k)): int(v) for k, v in skill_user_counts.items()},
        'skill_problem_counts': {str(int(k)): int(v) for k, v in skill_problem_counts.items()},
        'coverage_rate': len(valid_users) / df['user_id'].nunique()
    }
    
    # Create subdirectory based on skill IDs
    base_output_path = Path(output_dir)
    output_path = base_output_path / skill_id_str
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate config name if not provided
    if config_name is None:
        config_name = f'config_{skill_id_str}'
    
    # Remove .json extension if provided
    if config_name.endswith('.json'):
        config_name = config_name[:-5]
    
    # Save configuration file
    config_file = output_path / f'{config_name}.json'
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n{'='*80}")
    print("CONFIGURATION FILES SAVED")
    print(f"{'='*80}")
    print(f"✓ Configuration: {config_file}")
    
    # Save valid user IDs
    valid_users_df = pd.DataFrame({'user_id': sorted(valid_users)})
    users_file = output_path / f'valid_users_{skill_id_str}.csv'
    valid_users_df.to_csv(users_file, index=False)
    print(f"✓ Valid users: {users_file}")
    
    # Save verification report
    report_file = output_path / f'verification_{skill_id_str}.txt'
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("SKILL SET VERIFICATION REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Configuration: {config_name}\n")
        f.write(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Selected Skills ({K}) - Sorted by ID:\n")
        for i, skill_id in enumerate(sorted_skill_ids, 1):
            name = skill_names.get(skill_id, f"Skill {skill_id}")
            user_count = skill_user_counts.get(skill_id, 0)
            problem_count = skill_problem_counts.get(skill_id, 0)
            f.write(f"  {i:2d}. {name}\n")
            f.write(f"      ID: {skill_id}, Users: {user_count}, Problems: {problem_count}\n")
        f.write(f"\nValid Users: {len(valid_users)}\n")
        f.write(f"Coverage Rate: {config['coverage_rate']:.2%}\n")
        f.write("\n" + "="*80 + "\n")
    print(f"✓ Verification report: {report_file}")
    
    print(f"\n{'='*80}")
    print("NEXT STEPS")
    print(f"{'='*80}")
    print(f"\n1. Review the verification report to confirm data quality")
    print(f"\n2. Extract and process data:")
    print(f"   python scripts/assistments_2009_2010/extract_selected_data.py \\")
    print(f"     --config_name {config_name} \\")
    print(f"     --output_dir {output_path}")
    print(f"\n3. Estimate skill states:")
    print(f"   python scripts/assistments_2009_2010/estimate_skill_states.py \\")
    print(f"     --data_dir {output_path} \\")
    print(f"     --K {K}")
    print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Create skill set configuration with manually selected skills',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # By skill names:
  python create_manual_skillset.py --skills "Box and Whisker" "Circle Graph" "Venn Diagram"
  
  # By skill IDs:
  python create_manual_skillset.py --skills 280.0 70.0 77.0
  
  # Mixed (names and IDs):
  python create_manual_skillset.py --skills "Box and Whisker" 70.0 "Venn Diagram"
  
  # List all available skills:
  python create_manual_skillset.py --list_skills
        """
    )
    
    parser.add_argument(
        '--skills',
        type=str,
        nargs='+',
        help='List of skill names or IDs (space-separated). Use quotes for multi-word names.'
    )
    parser.add_argument(
        '--list_skills',
        action='store_true',
        help='Display all available skill names and IDs, then exit'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default='data/raw/assistments_2009_2010/skill_builder_data.csv',
        help='Path to raw data CSV (default: data/raw/assistments_2009_2010/skill_builder_data.csv)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/processed/assistments_2009_2010/selected_data',
        help='Output directory (default: data/processed/assistments_2009_2010/selected_data)'
    )
    parser.add_argument(
        '--config_name',
        type=str,
        default=None,
        help='Custom configuration name (default: auto-generated from skill IDs)'
    )
    parser.add_argument(
        '--description',
        type=str,
        default='Manually selected skills',
        help='Description for the configuration'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.list_skills and not args.skills:
        parser.error("Either --skills or --list_skills must be specified")
    
    create_manual_configuration(
        data_path=args.data_path,
        skill_identifiers=args.skills if args.skills else [],
        output_dir=args.output_dir,
        config_name=args.config_name,
        description=args.description,
        show_available_skills=args.list_skills
    )


if __name__ == '__main__':
    main()