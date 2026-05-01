import os
import time
import pandas as pd
import datetime
from typing import List, Dict
from corpus_loader import CorpusLoader
from retriever import Retriever
from agent import SupportAgent

# Rich imports
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich import box

def main():
    """
    Orchestrates the support triage process with a Rich terminal UI.
    """
    console = Console()
    
    # Configuration - paths are relative to project root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    
    input_paths = [
        os.path.join(base_dir, "support_tickets", "support_issues.csv"),
        os.path.join(base_dir, "support_tickets", "support_tickets.csv")
    ]
    input_csv = next((p for p in input_paths if os.path.exists(p)), None)
    output_csv = os.path.join(base_dir, "support_tickets", "output.csv")
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        console.print("[bold red]CRITICAL ERROR:[/bold red] GROQ_API_KEY environment variable not set.")
        return

    if not input_csv:
        console.print(f"[bold red]CRITICAL ERROR:[/bold red] Input CSV not found in support_tickets/.")
        return

    console.print(Panel("[bold cyan]Support Triage Agent v2.0[/bold cyan]\n[italic]Powered by Groq Llama-3 & Rich UI[/italic]", box=box.DOUBLE))
    
    # 1. Initialize Corpus
    with console.status("[bold yellow]Loading documentation corpus...[/bold yellow]"):
        loader = CorpusLoader(data_dir)
        documents = loader.load_corpus()
    console.print(f"[green]✓[/green] Successfully indexed [bold]{len(documents)}[/bold] documents.")

    # 2. Initialize Retrieval & Agent
    retriever = Retriever(documents)
    agent = SupportAgent(api_key)

    # 3. Read Tickets
    df = pd.read_csv(input_csv)
    required_cols = ['Issue', 'Subject', 'Company']
    for col in required_cols:
        if col not in df.columns:
            console.print(f"[bold red]ERROR:[/bold red] Missing required column '{col}' in input CSV.")
            return

    results = []
    
    # Setup for Live UI
    table = Table(title="Live Processing Feed", box=box.SIMPLE_HEAVY)
    table.add_column("#", justify="right", style="dim")
    table.add_column("Timestamp", style="cyan")
    table.add_column("Company", style="magenta")
    table.add_column("Status", justify="center")
    table.add_column("Product Area", style="yellow")
    table.add_column("Request Type", style="blue")

    # Progress bar
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        auto_refresh=False
    )
    task_id = progress.add_task("Triaging Tickets...", total=len(df))

    stats = {
        "replied": 0,
        "escalated": 0,
        "companies": {}
    }

    # 4. Process Loop with Live UI
    console.print(f"\n[bold]Starting triage for {len(df)} tickets...[/bold]")
    
    with Live(Panel(table, title="[bold blue]Real-time Triage Activity[/bold blue]"), console=console, refresh_per_second=4) as live:
        for index, row in df.iterrows():
            # Batch rate limiting: Sleep 60s every 10 tickets
            if index > 0 and index % 10 == 0:
                live.update(Panel(table, title=f"[bold red]RATE LIMIT SAFEGUARD: Sleeping 60s...[/bold red]"))
                time.sleep(60)
                live.update(Panel(table, title="[bold blue]Real-time Triage Activity[/bold blue]"))

            ticket = row.to_dict()
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            
            # Keyword-based context retrieval
            subject_q = str(ticket.get('Subject', '') or '')
            issue_q = str(ticket.get('Issue', '') or '')
            query = f"{subject_q} {issue_q}"
            context_results = retriever.search(query, top_k=3)
            confidence = retriever.get_confidence_score(context_results)
            
            # Agent Classification & Response
            triage_result = agent.process_ticket(ticket, context_results, confidence)
            
            # Merge and Update Stats
            final_record = {**ticket, **triage_result}
            results.append(final_record)
            
            # Statistics
            status = triage_result['status']
            stats[status] += 1
            company = ticket.get('Company', 'Unknown')
            stats["companies"][company] = stats["companies"].get(company, 0) + 1
            
            # Update Table
            status_style = "bold green" if status == "replied" else "bold red"
            table.add_row(
                str(index + 1),
                timestamp,
                company,
                f"[{status_style}]{status.upper()}[/{status_style}]",
                triage_result.get('product_area', 'N/A'),
                triage_result.get('request_type', 'N/A')
            )
            
            progress.advance(task_id)
            progress.refresh()

    # 5. Summary Display
    console.print("\n" + Panel("[bold green]Triage Complete![/bold green]", expand=False))
    
    # Final Stats Table
    summary_table = Table(title="Triage Summary Statistics", box=box.ROUNDED)
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="bold")
    
    summary_table.add_row("Total Tickets", str(len(df)))
    summary_table.add_row("Replied", f"[green]{stats['replied']}[/green]")
    summary_table.add_row("Escalated", f"[red]{stats['escalated']}[/red]")
    
    console.print(summary_table)

    # Company Breakdown
    company_table = Table(title="Breakdown by Company", box=box.ROUNDED)
    company_table.add_column("Company", style="magenta")
    company_table.add_column("Tickets", justify="right")
    
    for comp, count in stats["companies"].items():
        company_table.add_row(comp, str(count))
    
    console.print(company_table)

    # 6. Write Output
    output_df = pd.DataFrame(results)
    final_cols = ['Issue', 'Subject', 'Company', 'status', 'product_area', 'response', 'justification', 'request_type']
    
    if not output_df.empty:
        output_df = output_df.reindex(columns=final_cols)
        output_df.to_csv(output_csv, index=False)
        console.print(f"\n[bold green]✓[/bold green] Results exported to [bold cyan]{output_csv}[/bold cyan]")
    
    log_path = os.path.join(base_dir, "triage_agent.log")
    console.print(f"[dim]Logs available at: {log_path}[/dim]")

if __name__ == "__main__":
    main()
