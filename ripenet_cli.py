import os
import sys
import time
import requests
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint

# --- CONFIG ---
# Your deployed Hugging Face API URL
API_URL = "https://alexcj10-ripenet-backend.hf.space/predict"

console = Console()

def print_banner():
    banner = """
    [bold green]
    ██████╗ ██╗██████╗ ███████╗███╗   ██╗███████╗████████╗
    ██╔══██╗██║██╔══██╗██╔════╝████╗  ██║██╔════╝╚══██╔══╝
    ██████╔╝██║██████╔╝█████╗  ██╔██╗ ██║█████╗     ██║   
    ██╔══██╗██║██╔═══╝ ██╔══╝  ██║╚██╗██║██╔══╝     ██║   
    ██║  ██║██║██║     ███████╗██║ ╚████║███████╗   ██║   
    ╚═╝  ╚═╝╚═╝╚═╝     ╚══════╝╚═╝  ╚═══╝╚══════╝   ╚═╝   
    [/bold green]
    [bold white]Advanced AI Fruit Quality Analysis Suite (Cloud Powered)[/bold white]
    """
    console.print(banner)

def scan_image(image_path):
    if not os.path.exists(image_path):
        console.print(f"[bold red]❌ Error: File not found at {image_path}[/bold red]")
        return None

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description="[cyan]🛰️ Sending image to RipeNet Cloud AI...", total=None)
        
        try:
            with open(image_path, "rb") as f:
                files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
                response = requests.post(API_URL, files=files, timeout=30)
                
            if response.status_code == 200:
                try:
                    data = response.json()
                    if isinstance(data, dict):
                        # Prioritize the formatted report from the API
                        if data.get("report"):
                            return str(data["report"])
                        
                        # Fallback mapping (Handles various key names for robustness)
                        fruit = data.get("fruit") or data.get("fruit_type", "Unknown")
                        ripeness = data.get("ripeness") or data.get("status", "Unknown")
                        shelf = data.get("shelf_life_days") or data.get("shelf_life") or data.get("days", "??")
                        
                        return f"The model detects a {str(ripeness).lower()} {str(fruit).lower()}. Estimated shelf life: {shelf} days."
                    return str(data).strip()
                except Exception as je:
                    console.print(f"[dim]Raw Response: {response.text}[/dim]")
                    return f"Error parsing response: {je}"
            else:
                console.print(f"[bold red]❌ API Error ({response.status_code}): {response.text}[/bold red]")
                return None
        except Exception as e:
            console.print(f"[bold red]❌ Connection Error: {str(e)}[/bold red]")
            return None

def batch_process(directory):
    if not os.path.isdir(directory):
        console.print(f"[bold red]❌ Error: {directory} is not a directory[/bold red]")
        return

    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not files:
        console.print("[yellow]⚠️ No image files found in directory.[/yellow]")
        return

    table = Table(title=f"RipeNet Batch Analysis: {os.path.basename(directory)}")
    table.add_column("File Name", style="cyan")
    table.add_column("AI Analysis", style="white")
    table.add_column("Status", style="bold")

    with Progress() as progress:
        task = progress.add_task("[green]🛰️ Batch Scanning (Cloud)...", total=len(files))
        
        for file in files:
            result = scan_image(file)
            if result:
                status = "[green]FRESH[/green]" if "fresh" in result.lower() else "[yellow]UNRIPE[/yellow]" if "unripe" in result.lower() else "[red]ROTTEN[/red]"
                table.add_row(os.path.basename(file), result, status)
            else:
                table.add_row(os.path.basename(file), "[red]FAILED[/red]", "[dim]ERROR[/dim]")
            progress.update(task, advance=1)

    console.print(table)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="🍇 RipeNet Global CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Scan command
    scan_parser = subparsers.add_parser("scan", help="Analyze a single image via Cloud AI")
    scan_parser.add_argument("image", help="Path to the fruit image")

    # Batch command
    batch_parser = subparsers.add_parser("batch", help="Analyze a folder via Cloud AI")
    batch_parser.add_argument("dir", help="Path to the images directory")

    # Info command
    subparsers.add_parser("info", help="Show system and API status")

    args = parser.parse_args()

    # Pre-check for info
    if args.command == "info":
        print_banner()
        table = Table(title="RipeNet Cloud Status")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("API Endpoint", API_URL)
        table.add_row("Mode", "🛰️ Cloud Inference")
        
        try:
            start = time.time()
            # Simple ping check if possible, or just HEAD
            resp = requests.get(API_URL.replace("/predict", "/"), timeout=5)
            latency = f"{(time.time() - start)*1000:.0f}ms"
            table.add_row("API Status", "✅ Online" if resp.status_code == 200 else "⚠️ Issues")
            table.add_row("Latency", latency)
        except:
            table.add_row("API Status", "❌ Offline")

        console.print(table)
        console.print(f"\n[dim]Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]")
        return

    if not args.command:
        parser.print_help()
        return

    print_banner()

    if args.command == "scan":
        result = scan_image(args.image)
        if result:
            color = "green" if "fresh" in result.lower() else "yellow" if "unripe" in result.lower() else "red"
            console.print(Panel(
                f"[bold {color}]{result}[/bold {color}]",
                title="[bold white]AI Analysis (Cloud Result)[/bold white]",
                border_style=color,
                padding=(1, 2)
            ))
            
    elif args.command == "batch":
        batch_process(args.dir)

if __name__ == "__main__":
    main()
