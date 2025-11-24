"""
Run All Ablation Studies
=========================

Master script to execute all ablation experiments sequentially.
Provides progress tracking and consolidated reporting.
"""

import subprocess
import time
import sys
from datetime import datetime

def run_experiment(script_name, description):
    """Run a single ablation experiment"""
    print("\n" + "="*80)
    print(f"🔬 Starting: {description}")
    print(f"   Script: {script_name}")
    print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            check=True
        )
        
        elapsed = time.time() - start_time
        
        print(result.stdout)
        
        print("\n" + "="*80)
        print(f"✅ Completed: {description}")
        print(f"   Duration: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        print("="*80)
        
        return {
            'name': description,
            'script': script_name,
            'status': 'SUCCESS',
            'duration': elapsed,
            'output': result.stdout
        }
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        
        print(f"\n❌ Error in {description}:")
        print(e.stdout)
        print(e.stderr)
        
        return {
            'name': description,
            'script': script_name,
            'status': 'FAILED',
            'duration': elapsed,
            'error': str(e)
        }

def main():
    print("="*80)
    print("🔬 ABLATION STUDY SUITE")
    print("   Factorized Categorical HMM for ENSO Detection")
    print("="*80)
    print(f"\nStart Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nThis will run 2 comprehensive ablation experiments:")
    print("  1. Feature Ablation (9 configurations)")
    print("  2. Temporal Dependency Ablation (2 models)")
    print("\nEstimated total time: 45-60 minutes")
    print("="*80)
    
    experiments = [
        ('feature_ablation.py', 'Feature Ablation Study'),
        ('temporal_ablation.py', 'Temporal Dependency Ablation'),
    ]
    
    results = []
    overall_start = time.time()
    
    for script, description in experiments:
        result = run_experiment(script, description)
        results.append(result)
        
        # Brief pause between experiments
        time.sleep(2)
    
    overall_elapsed = time.time() - overall_start
    
    # Summary Report
    print("\n\n" + "="*80)
    print("📊 ABLATION STUDY SUMMARY")
    print("="*80)
    
    print(f"\nTotal Duration: {overall_elapsed:.1f} seconds ({overall_elapsed/60:.1f} minutes)")
    print(f"Completion Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n" + "-"*80)
    print("Experiment Results:")
    print("-"*80)
    
    for i, result in enumerate(results, 1):
        status_icon = "✅" if result['status'] == 'SUCCESS' else "❌"
        print(f"\n{i}. {status_icon} {result['name']}")
        print(f"   Status: {result['status']}")
        print(f"   Duration: {result['duration']:.1f}s ({result['duration']/60:.1f}m)")
        if result['status'] == 'FAILED':
            print(f"   Error: {result.get('error', 'Unknown error')}")
    
    # Count successes
    successes = sum(1 for r in results if r['status'] == 'SUCCESS')
    failures = len(results) - successes
    
    print("\n" + "-"*80)
    print(f"Summary: {successes}/{len(results)} experiments completed successfully")
    if failures > 0:
        print(f"⚠️  {failures} experiment(s) failed - check logs above")
    else:
        print("🎉 All experiments completed successfully!")
    print("-"*80)
    
    # List output files
    print("\n" + "="*80)
    print("📁 Generated Output Files:")
    print("="*80)
    
    output_files = [
        "\n1. Feature Ablation:",
        "   - feature_ablation_results.csv",
        "   - feature_ablation_analysis.png",
        "\n2. Temporal Ablation:",
        "   - temporal_ablation_results.csv",
        "   - temporal_ablation_analysis.png",
    ]
    
    for line in output_files:
        print(line)
    
    print("\n" + "="*80)
    print("🎓 Next Steps:")
    print("="*80)
    print("""
1. Review the generated CSV files for detailed metrics
2. Examine the PNG visualizations for insights
3. Compare results across experiments
4. Update main README with key findings
5. Consider additional ablation studies based on results
    """)
    
    print("="*80)
    print("Ablation Study Suite Complete!")
    print("="*80)
    
    return successes == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

