
CONFIG_DIR="configs/single-runs/new-encoder-decoder"
LOG_DIR="logs"

config_prefix="$(echo "$CONFIG_DIR" | sed 's#[/ ]#_#g')"
find "$CONFIG_DIR" -type f \( -name "*.yaml" -o -name "*.yml" \) | while read -r yaml_file; do
    base_name="$(basename "$yaml_file")"
    log_file="$LOG_DIR/${config_prefix}_${base_name%.*}.log"

    echo "Training with $yaml_file"
    echo "Writing output to: $log_file"
    python train.py --experiment "$yaml_file" > "$log_file" 2>&1 | tee "$log_file"
done