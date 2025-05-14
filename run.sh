#!/bin/bash

# Default values
BASE_PATH="/workspaces/federated-learning-dropout"
ROOT_DATA_PATH=""
LABEL_DATA_PATH=""

run_single_experiment() {
    exp_name=$1
    cmd="python main.py experiment=$exp_name"
    
    if [ -n "$BASE_PATH" ]; then
        cmd="$cmd base_path=$BASE_PATH"
    fi
    
    if [ -n "$DATA" ]; then
        cmd="$cmd data=$DATA"
    fi

    # Add data_path override if specified
    if [ -n "$ROOT_DATA_PATH" ]; then
        cmd="$cmd data.root_path=$ROOT_DATA_PATH"
    fi
    if [ -n "$LABEL_DATA_PATH" ]; then
        cmd="$cmd data.label_path=$LABEL_DATA_PATH"
    fi

    if [ -n "$WANDB_API_KEY" ]; then
        cmd="$cmd tracking.api_key=$WANDB_API_KEY"
    fi

    if [ -n "$NUM_GPU" ]; then
        cmd="$cmd gpus=$NUM_GPU"
    fi

    if [ -n "$MODEL" ]; then
        cmd="$cmd model=$MODEL"
    fi

    if [ -n "$NUM_CLIENTS" ]; then
        cmd="$cmd num_clients=$NUM_CLIENTS"
    fi

    if [ -n "$ROUND" ]; then
        cmd="$cmd num_rounds=$ROUND"
    fi


    
    echo "Running experiment: $exp_name"
    echo "Command: $cmd"
    eval $cmd
}

run_all_experiments() {
    echo "Running all experiments sequentially..."
    cmd="python main.py -m experiment=random_30pct,alternate_dropout,fixed_30pct,random_70pct"
    
    # Add base_path override if specified
    if [ -n "$BASE_PATH" ]; then
        cmd="$cmd base_path=$BASE_PATH"
    fi

    if [ -n "$DATA" ]; then
        cmd="$cmd data=$DATA"
    fi
    
    # Add data_path overrides if specified
    if [ -n "$ROOT_DATA_PATH" ]; then
        cmd="$cmd data.root_path=$ROOT_DATA_PATH"
    fi
    if [ -n "$LABEL_DATA_PATH" ]; then
        cmd="$cmd data.label_path=$LABEL_DATA_PATH"
    fi
    if [ -n "$WANDB_API_KEY" ]; then
        cmd="$cmd tracking.api_key=$WANDB_API_KEY"
    fi
    if [ -n "$NUM_GPU" ]; then
        cmd="$cmd gpus=$NUM_GPU"
    fi

    if [ -n "$MODEL" ]; then
        cmd="$cmd model=$MODEL"
    fi

    if [ -n "$NUM_CLIENTS" ]; then
        cmd="$cmd num_clients=$NUM_CLIENTS"
    fi

    if [ -n "$ROUND" ]; then
        cmd="$cmd num_rounds=$ROUND"
    fi

    
    echo "Command: $cmd"
    eval $cmd
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -a|--all)
            RUN_ALL=true
            shift
            ;;
        -e|--experiment)
            if [ -z "$2" ]; then
                echo "Please provide an experiment name."
                exit 1
            fi
            EXPERIMENT="$2"
            shift 2
            ;;
        -b|--base-path)
            if [ -z "$2" ]; then
                echo "Please provide a base path."
                exit 1
            fi
            BASE_PATH="$2"
            shift 2
            ;;
        -r|--root-data-path)
            if [ -z "$2" ]; then
                echo "Please provide a root data path."
                exit 1
            fi
            ROOT_DATA_PATH="$2"
            shift 2
            ;;
        -l|--label-data-path)
            if [ -z "$2" ]; then
                echo "Please provide a label data path."
                exit 1
            fi
            LABEL_DATA_PATH="$2"
            shift 2
            ;;

        -w | --wandb)
            if [ -z "$2" ]; then
                echo "Please provide a wandb api key."
                exit 1
            fi
            WANDB_API_KEY="$2"
            shift 2
            ;;

        -n | --num-gpu)
            if [ -z "$2" ]; then
                echo "Please provide a number of gpus."
                exit 1
            fi
            NUM_GPU="$2"
            shift 2
            ;;

        -m | --model)
            if [ -z "$2" ]; then
                echo "Please provide a model."
                exit 1
            fi
            MODEL="$2"
            shift 2
            ;;

        -c | --num-clients)
            if [ -z "$2" ]; then
                echo "Please provide a number of clients."
                exit 1
            fi
            NUM_CLIENTS="$2"
            shift 2
            ;;
        -ro | --num_rounds)
            if [ -z "$2" ]; then
                echo "Please provide a number of rounds."
                exit 1
            fi
            ROUND="$2"
            shift 2
            ;;

        -d | --data)
            if [ -z "$2" ]; then
                echo "Please provide a data path."
                exit 1
            fi
            DATA="$2"
            shift 2
            ;;

        -h|--help)
            echo "Usage: $0 [-a | --all] | [-e | --experiment <experiment_name>] | [-b | --base-path <path>] | [-r | --root-data-path <path>] | [-l | --label-data-path <path>] | [-h | --help]"
            echo "  -a, --all                Run all experiments sequentially."
            echo "  -e, --experiment         Run a specific experiment."
            echo "  -b, --base-path          Override the base path (default: /workspaces/federated-learning-dropout)."
            echo "  -r, --root-data-path     Override the root data path."
            echo "  -l, --label-data-path    Override the label data path."
            echo "  -h, --help               Show this help message."
            exit 0
            ;;
        *)
            echo "Invalid option: $1. Use -h or --help for usage information."
            exit 1
            ;;
    esac
done

# Run the specified experiment(s)
if [ "$RUN_ALL" = true ]; then
    run_all_experiments
elif [ -n "$EXPERIMENT" ]; then
    run_single_experiment "$EXPERIMENT"
else
    echo "Please specify either -a/--all to run all experiments or -e/--experiment to run a specific experiment."
    echo "Use -h or --help for usage information."
    exit 1
fi

echo "Experiment completed."