import os
import time
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple

class AgentVisualizer:
    """
    Utility for visualizing agent interactions and API calls.
    Generates visual diagrams showing the flow of information between components.
    Still works with the Responses API implementation by tracking steps rather than agent interactions.
    """

    def __init__(self, log_dir=None):
        """
        Initialize the agent visualizer

        Args:
            log_dir: Directory to save visualization files (default: 'logs/visuals')
        """
        self.log_dir = log_dir or Path('logs/visuals')
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Tracking structures
        self.agent_calls = []
        self.api_calls = []
        self.handoffs = []
        self.token_usage = {}

    def reset(self):
        """Reset tracking data"""
        self.agent_calls = []
        self.api_calls = []
        self.handoffs = []
        self.token_usage = {}

    def track_agent_call(self, agent_name, input_text, output_text, duration, tokens_used=None):
        """
        Track an agent call or component step

        Args:
            agent_name: Name of the agent or component
            input_text: Input text to the agent
            output_text: Output text from the agent
            duration: Duration of the call in seconds
            tokens_used: Dictionary of token usage by model
        """
        self.agent_calls.append({
            'agent': agent_name,
            'timestamp': time.time(),
            'input_length': len(input_text),
            'output_length': len(output_text),
            'duration': duration,
            'tokens': tokens_used
        })

        # Update token usage
        if tokens_used:
            for model, tokens in tokens_used.items():
                if model not in self.token_usage:
                    self.token_usage[model] = 0
                self.token_usage[model] += tokens

    def track_api_call(self, function_name, api_type, success, duration):
        """
        Track an API call

        Args:
            function_name: Name of the function called
            api_type: Type of API call (e.g., 'chat.completions', 'responses')
            success: Whether the call was successful
            duration: Duration of the call in seconds
        """
        self.api_calls.append({
            'function': function_name,
            'type': api_type,
            'timestamp': time.time(),
            'success': success,
            'duration': duration
        })

    def track_handoff(self, from_agent, to_agent, input_text):
        """
        Track an information handoff between components

        Args:
            from_agent: Name of the component sending information
            to_agent: Name of the component receiving the information
            input_text: Text passed in the handoff
        """
        self.handoffs.append({
            'from': from_agent,
            'to': to_agent,
            'timestamp': time.time(),
            'text_length': len(input_text)
        })

    def generate_flow_diagram(self, title="Component Interaction Flow", output_file=None):
        """
        Generate a flow diagram showing component interactions

        Args:
            title: Title for the diagram
            output_file: Output file path (default: logs/visuals/flow_{timestamp}.png)

        Returns:
            Path to the generated diagram
        """
        if not output_file:
            timestamp = int(time.time())
            output_file = self.log_dir / f"flow_{timestamp}.png"

        # Create a directed graph
        G = nx.DiGraph()

        # Add nodes for each component (deduplicate)
        agents = {call['agent'] for call in self.agent_calls}
        for handoff in self.handoffs:
            agents.add(handoff['from'])
            agents.add(handoff['to'])

        # Add nodes to the graph
        for agent in agents:
            G.add_node(agent)

        # Add edges for handoffs
        for handoff in self.handoffs:
            # Check if edge already exists
            if G.has_edge(handoff['from'], handoff['to']):
                # Increment weight
                G[handoff['from']][handoff['to']]['weight'] += 1
            else:
                # Create new edge with weight 1
                G.add_edge(handoff['from'], handoff['to'], weight=1)

        # Set up the figure
        plt.figure(figsize=(12, 8))

        # Use a hierarchical layout
        pos = nx.spring_layout(G, seed=42)

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=3000, alpha=0.8)

        # Draw edges with varying width based on weight
        edge_width = [G[u][v]['weight'] * 2 for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, width=edge_width, alpha=0.7, edge_color='gray',
                              arrows=True, arrowsize=20, arrowstyle='-|>')

        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=12, font_family='sans-serif')

        # Draw edge labels (number of handoffs)
        edge_labels = {(u, v): f"{G[u][v]['weight']} handoffs" for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

        # Add title and other details
        plt.title(title, fontsize=16)

        # Add token usage information
        if self.token_usage:
            token_text = "Token Usage:\n"
            for model, tokens in self.token_usage.items():
                token_text += f"{model}: {tokens:,} tokens\n"
            plt.figtext(0.02, 0.02, token_text, fontsize=10)

        # Add timestamp
        timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        plt.figtext(0.98, 0.02, f"Generated: {timestamp_str}", fontsize=8, ha='right')

        # Remove axes
        plt.axis('off')

        # Save the figure
        plt.tight_layout()
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
        plt.close()

        return output_file

    def generate_timeline_diagram(self, title="Execution Timeline", output_file=None):
        """
        Generate a timeline diagram showing when components were called

        Args:
            title: Title for the diagram
            output_file: Output file path

        Returns:
            Path to the generated diagram
        """
        if not output_file:
            timestamp = int(time.time())
            output_file = self.log_dir / f"timeline_{timestamp}.png"

        # Get all component names
        agent_names = {call['agent'] for call in self.agent_calls}

        # Create figure
        fig, ax = plt.subplots(figsize=(15, 8))

        # Assign y-positions to components
        agent_positions = {agent: i for i, agent in enumerate(sorted(agent_names))}

        # Set the first call time as time zero
        if self.agent_calls:
            t0 = min(call['timestamp'] for call in self.agent_calls)
        else:
            t0 = time.time()

        # Draw timeline bars for each component call
        colors = plt.cm.tab10.colors

        for i, call in enumerate(sorted(self.agent_calls, key=lambda x: x['timestamp'])):
            agent = call['agent']
            start_time = call['timestamp'] - t0
            duration = call['duration']
            color = colors[agent_positions[agent] % len(colors)]

            # Draw bar representing call duration
            ax.barh(agent_positions[agent], duration, left=start_time, height=0.5,
                   color=color, alpha=0.7, label=agent if i == 0 else "")

            # Add text for token usage
            if call.get('tokens'):
                total_tokens = sum(call['tokens'].values())
                if total_tokens > 0:
                    ax.text(start_time + duration/2, agent_positions[agent],
                           f"{total_tokens:,} tokens",
                           ha='center', va='center', fontsize=8, color='black')

        # Draw handoff arrows
        for handoff in self.handoffs:
            from_agent = handoff['from']
            to_agent = handoff['to']
            time_point = handoff['timestamp'] - t0

            # Draw arrow
            arrow = FancyArrowPatch(
                (time_point, agent_positions[from_agent]),
                (time_point, agent_positions[to_agent]),
                arrowstyle='-|>', mutation_scale=15,
                color='red', linewidth=1.5, alpha=0.7
            )
            ax.add_patch(arrow)

        # Set labels and title
        ax.set_yticks(list(agent_positions.values()))
        ax.set_yticklabels(list(agent_positions.keys()))
        ax.set_xlabel('Time (seconds)')
        ax.set_title(title)

        # Set grid
        ax.grid(True, axis='x', linestyle='--', alpha=0.3)

        # Add timestamp
        timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        plt.figtext(0.98, 0.02, f"Generated: {timestamp_str}", fontsize=8, ha='right')

        # Save the figure
        plt.tight_layout()
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
        plt.close()

        return output_file

    def generate_report(self, output_file=None):
        """
        Generate a JSON report with all tracked data

        Args:
            output_file: Output file path

        Returns:
            Path to the generated report
        """
        if not output_file:
            timestamp = int(time.time())
            output_file = self.log_dir / f"report_{timestamp}.json"

        report = {
            'timestamp': time.time(),
            'agent_calls': self.agent_calls,
            'api_calls': self.api_calls,
            'handoffs': self.handoffs,
            'token_usage': self.token_usage,
            'summary': {
                'total_components': len({call['agent'] for call in self.agent_calls}),
                'total_calls': len(self.agent_calls),
                'total_handoffs': len(self.handoffs),
                'total_api_calls': len(self.api_calls),
                'total_tokens': sum(self.token_usage.values()) if self.token_usage else 0
            }
        }

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        return output_file

# Singleton instance
_visualizer_instance = None

def get_visualizer():
    """Get the visualizer singleton instance"""
    global _visualizer_instance
    if _visualizer_instance is None:
        _visualizer_instance = AgentVisualizer()
    return _visualizer_instance

def initialize_visualizer(log_dir=None):
    """Initialize the visualizer singleton"""
    global _visualizer_instance
    _visualizer_instance = AgentVisualizer(log_dir)
    return _visualizer_instance
