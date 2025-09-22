# Define parameter ranges
$K_values = @(5, 10, 20)
$tasks = @(1, 2, 3, 5)
$datasets = @("arguana", "kialo", "opinionqa")

# Loop through all combinations
foreach ($dataset in $datasets) {
    foreach ($K in $K_values) {
        foreach ($task in $tasks) {

            Write-Host "Running experiment: Dataset=$dataset, K=$K, Task=$task"
            
            python .\eval.py --dataset $dataset --K $K --task $task
            
            Start-Sleep -Seconds 1
        }
    }
}

$lambd_values = @(0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0)

foreach ($dataset in $datasets) {
    foreach ($K in $K_values) {
        foreach ($lambd in $lambd_values) {
            Write-Host "Running experiment with lambda=$lambd"

            python .\eval.py --dataset $dataset --K $K --task 4 --lambd $lambd --val

            Start-Sleep -Seconds 1
        }
    }
}
$lambds_chosen = @{
    "arguana" = 0.4
    "kialo" = 0.4
    "opinionqa" = 0.5
}

foreach ($dataset in $datasets) {
    foreach ($K in $K_values) {
        $lambd = $lambds_chosen[$dataset]
        Write-Host "Running experiment: Dataset=$dataset, K=$K, Lambda=$lambd"
        python .\eval.py --dataset $dataset --K $K --task 4 --lambd $lambd
        Start-Sleep -Seconds 1
    }
}

Write-Host "All experiments completed."