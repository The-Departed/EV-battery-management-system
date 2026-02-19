#!/bin/bash
# Quick commands for Battery Modeling Project

echo "🔋 Battery Core Temperature Estimation - Quick Commands"
echo "========================================================"
echo ""

# Function to show menu
show_menu() {
    echo "Available commands:"
    echo ""
    echo "  1) Load data (Step 1.1)"
    echo "  2) Generate visualizations"
    echo "  3) Show summary"
    echo "  4) Download dataset (if needed)"
    echo "  5) Check project status"
    echo "  6) Exit"
    echo ""
}

# Main loop
while true; do
    show_menu
    read -p "Select option (1-6): " choice
    
    case $choice in
        1)
            echo "Loading B0005 battery data..."
            uv run python ecm/data_loader.py
            ;;
        2)
            echo "Generating visualizations..."
            uv run python ecm/visualize.py
            ;;
        3)
            echo "Showing Step 1.1 summary..."
            uv run python ecm/step1_1_summary.py
            ;;
        4)
            echo "Downloading NASA dataset..."
            ./scripts/download_data.sh
            ;;
        5)
            echo "Project status:"
            cat PROJECT_STATUS.md | head -50
            ;;
        6)
            echo "Exiting..."
            break
            ;;
        *)
            echo "Invalid option. Please select 1-6."
            ;;
    esac
    
    echo ""
    read -p "Press Enter to continue..."
    clear
done
