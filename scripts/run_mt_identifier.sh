
#!bin/bash
# This script is used to run the pipeline
# INPUT_JSON=/afs/cern.ch/work/d/dapullia/public/dune/machine_learning/json/regression/hp_classification.json
# INPUT_JSON=/afs/cern.ch/work/d/dapullia/public/dune/machine_learning/json/mt_identification/basic-hp_identification.json

#OUTPUT_FOLDER=/eos/user/h/hakins/dune/ML/mt_identifier/
# Parse command-line arguments 
while [[ $# -gt 0 ]]; do
    case "$1" in
        -i|--input_json)
            INPUT_JSON="$2"
            shift 2
            ;;
        -o|--output_folder)
            OUTPUT_FOLDER="$2"
            shift 2
            ;;
            
        -cut|--cut)
            cut="$2"
            shift 2
            ;;

        -h|--help)
            print_help
            ;;
        *)
            shift
            ;;
    esac
done
output_file="/afs/cern.ch/work/h/hakins/private/ml_for_pointing/json/mt_identification/basic-hp_identification_${cut}.json"
cd /afs/cern.ch/work/h/hakins/private/online-pointing-utils/scripts/json_creators
python hp_identification_json_creator.py --cut "$cut" --output_file "$output_file"
INPUT_JSON="$output_file"

REPO_HOME=$(git rev-parse --show-toplevel)
export PYTHONPATH=$PYTHONPATH:$REPO_HOME/custom_python_libs/lib/python3.9/site-packages

cd /afs/cern.ch/work/h/hakins/private/ml_for_pointing/mt_identifier
python main.py --input_json $INPUT_JSON --output_folder $OUTPUT_FOLDER
cd ../scripts
