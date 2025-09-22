# Define parameter ranges
$K_values = @(5, 10, 20)
# $tasks = @(1, 2, 3, 4)
$tasks = @(5)
$datasets = @("arguana", "kialo", "opinionqa")
$lambd = 0.5

# Loop through all combinations
foreach ($dataset in $datasets) {
    foreach ($K in $K_values) {
        foreach ($task in $tasks) {

            Write-Host "Running experiment: Dataset=$dataset, K=$K, Task=$task"
            
            python .\tasks.py --dataset $dataset --K $K --task $task --lambd $lambd
            
            Start-Sleep -Seconds 1
        }
    }
}

# $lambd_values = @(0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0)

# foreach ($K in $K_values) {
#     foreach ($lambd in $lambd_values) {
#         Write-Host "Running experiment with lambda=$lambd"
#         # Run the Python script with the current parameters
#         python .\tasks.py --dataset kialo --K $K --task 4 --lambd $lambd
#     }
# }

Write-Host "All experiments completed."
# python .\tasks.py --dataset arguana --K 5 --task 3 --lambd 0.5