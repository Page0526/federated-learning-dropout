run_single_experiment() {
    exp_name=$1
    echo "Running experiment: $exp_name"
    python main.py experiment=$exp_name
}

run_all_experiments() {
    echo "Running all experiments sequentially..."
    python main.py -m experiment=random_30pct,alternate,fixed_30pct,random_70pct,random_with_fixed
}

case "$1" in 
    -a | -all) 
        run_all_experiments
        ;;
    -e | -experiment)
        if [ -z "$2" ]; then
            echo "Please provide an experiment name."
            exit 1
        fi
        run_single_experiment "$2"
        ;;
    -h | -help)
        echo "Usage: $0 [-a | -all] | [-e | -experiment <experiment_name>] | [-h | -help]"
        echo "  -a, -all         Run all experiments sequentially."
        echo "  -e, -experiment  Run a specific experiment."
        echo "  -h, -help        Show this help message."
        ;;
    *)
        echo "Invalid option. Use -h or --help for usage information."
        exit 1
        ;;
esac

echo "Experiment completed."
