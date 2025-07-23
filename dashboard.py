"""Streamlit dashboard for analyzing steered generation results."""

import json
from pathlib import Path
from typing import Dict, List, Any, Tuple

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import torch as th
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from plotly.subplots import make_subplots
from nnterp import StandardizedTransformer
from dct_diff import SteerSingleModel


class AppError(Exception):
    """Custom exception for dashboard errors."""
    pass


def error_and_raise(message: str):
    """Display error in UI and raise exception for command line."""
    st.error(message)
    raise AppError(message)

# Constants
MAX_COMPLETION_LENGTH = 200
STEERING_FACTOR_MIN = -3.0
STEERING_FACTOR_MAX = 3.0
STEERING_FACTOR_STEP = 0.1
DEFAULT_STEERING_FACTOR = 1.0


class SteeringDashboard:
    """Dashboard for analyzing and interacting with steered model generations."""
    
    def __init__(self):
        self.global_state_file = Path(".app_state")
        self.state_file = "app_state.json"
        self.initialize_session_state()
        self.load_global_state()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        defaults = {
            'selected_model': None,
            'selected_method': None, 
            'selected_file': None,
            'data': None,
            'steering_factors': {},
            'steering_types': {},
            'selected_steering_vectors': [],
            'expanded_completions': set(),
            'loaded_model': None,
            'steering_vectors': None,
            'current_exp_path': None
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def load_global_state(self):
        """Load global state from .app_state file."""
        if self.global_state_file.exists():
            try:
                with open(self.global_state_file, 'r') as f:
                    global_state = json.load(f)
                
                # Only load if not already set in session state
                for key in ['selected_model', 'selected_method', 'selected_file']:
                    if key in global_state and st.session_state.get(key) is None:
                        st.session_state[key] = global_state[key]
            except (json.JSONDecodeError, IOError):
                # If file is corrupted or unreadable, ignore and use defaults
                pass
    
    def save_global_state(self):
        """Save current selection state to global .app_state file."""
        try:
            # Create directory if it doesn't exist
            self.global_state_file.parent.mkdir(parents=True, exist_ok=True)
            
            global_state = {
                'selected_model': st.session_state.selected_model,
                'selected_method': st.session_state.selected_method,
                'selected_file': st.session_state.selected_file
            }
            
            with open(self.global_state_file, 'w') as f:
                json.dump(global_state, f, indent=2)
        except IOError:
            # If we can't write, silently continue
            pass
    
    def get_path(self, model_name: str, method: str) -> Dict[str, Any]:
        """Get path information for a given model and method."""
        results_dir = Path("results") / method / model_name
        if not results_dir.exists():
            error_and_raise(f"Results directory {results_dir} does not exist")
        
        exp_paths = []
        for exp_dir in results_dir.iterdir():
            if exp_dir.is_dir():
                json_file = exp_dir / "steering_data.json"
                pt_file = exp_dir / "steering_vectors.pt"
                if json_file.exists() and pt_file.exists():
                    exp_paths.append(exp_dir)
        
        if not exp_paths:
            error_and_raise(f"No complete experiment directories found in {results_dir}")
        
        return {
            "experiment_paths": exp_paths,
            "save_path": results_dir
        }
    
    def get_available_models_and_methods(self) -> Tuple[List[str], Dict[str, List[str]]]:
        """Get available models and their methods from the results directory."""
        results_dir = Path("results")
        if not results_dir.exists():
            error_and_raise("Results directory does not exist")
        
        models = set()
        methods = set()
        
        for method_dir in results_dir.iterdir():
            if method_dir.is_dir():
                method_name = method_dir.name
                methods.add(method_name)
                
                for model_dir in method_dir.iterdir():
                    if model_dir.is_dir():
                        model_name = model_dir.name
                        models.add(model_name)
        
        models = sorted(list(models))
        methods_per_model = {model: sorted(list(methods)) for model in models}
        
        return models, methods_per_model
    
    def load_data(self, file_path: Path) -> Dict[str, Any]:
        """Load experiment data from JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return self.validate_data_structure(data)
    
    def validate_data_structure(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that data has expected structure."""
        if 'baseline_generations' not in data:
            error_and_raise("Missing required key: baseline_generations")
        if not isinstance(data['baseline_generations'], dict):
            error_and_raise("baseline_generations must be a dictionary")
        if 'steered_generation' not in data:
            error_and_raise("Missing required key: steered_generation")
        if not isinstance(data['steered_generation'], list):
            error_and_raise("steered_generation must be a list")
        if 'steering_factors' not in data:
            error_and_raise("Missing required key: steering_factors")
        if not isinstance(data['steering_factors'], list):
            error_and_raise("steering_factors must be a list")
        if 'config' not in data:
            error_and_raise("Missing required key: config")
        if not isinstance(data['config'], dict):
            error_and_raise("config must be a dictionary")
        if 'exp_id' not in data:
            error_and_raise("Missing required key: exp_id")
        if 'model_name' not in data:
            error_and_raise("Missing required key: model_name")
        if 'closest_tokens' not in data:
            error_and_raise("Missing required key: closest_tokens")
        if 'median_norm' not in data:
            error_and_raise("Missing required key: median_norm")
        if 'rel_steering_factors' not in data:
            error_and_raise("Missing required key: rel_steering_factors")
        return data
    
    def save_state(self, save_path: Path):
        """Save current dashboard state to app_state.json."""
        state = {
            "selected_model": st.session_state.selected_model,
            "selected_method": st.session_state.selected_method,
            "selected_file": st.session_state.selected_file,
            "steering_factors": st.session_state.steering_factors,
            "selected_steering_vectors": st.session_state.selected_steering_vectors,
        }
        
        state_path = save_path / self.state_file
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)
        st.success("State saved successfully!")
    
    def load_state(self, save_path: Path):
        """Load dashboard state from app_state.json."""
        state_path = save_path / self.state_file
        if state_path.exists():
            with open(state_path, 'r') as f:
                state = json.load(f)
            
            for key, value in state.items():
                if key in st.session_state:
                    st.session_state[key] = value
    
    def create_sidebar(self):
        """Create the dashboard sidebar for model/method selection."""
        st.sidebar.title("Steering Analysis Dashboard")
        
        models, methods_per_model = self.get_available_models_and_methods()
        
        # Model selection
        if models:
            selected_model = st.sidebar.selectbox(
                "Select Model",
                options=models,
                index=models.index(st.session_state.selected_model) if st.session_state.selected_model in models else 0
            )
            st.session_state.selected_model = selected_model
            
            # Save global state when selection changes
            self.save_global_state()
            
            # Method selection
            if selected_model in methods_per_model:
                methods = methods_per_model[selected_model]
                if methods:
                    selected_method = st.sidebar.selectbox(
                        "Select Method",
                        options=methods,
                        index=methods.index(st.session_state.selected_method) if st.session_state.selected_method in methods else 0
                    )
                    st.session_state.selected_method = selected_method
                    
                    # Save global state when selection changes
                    self.save_global_state()
                    
                    # Experiment selection
                    path_info = self.get_path(selected_model, selected_method)
                    exp_paths = path_info["experiment_paths"]
                    
                    if exp_paths:
                        exp_names = [f.name for f in exp_paths]
                        # Create display names in slug_number format
                        display_names = []
                        for name in exp_names:
                            parts = name.split('_')
                            if len(parts) >= 2:
                                # Reverse order: put slug first, then number
                                slug = '_'.join(parts[1:])  # Everything after first underscore
                                number = parts[0]  # First part is the number
                                display_names.append(f"{slug}_{number}")
                            else:
                                display_names.append(name)  # Fallback if format is unexpected
                        
                        selected_display_name = st.sidebar.selectbox(
                            "Select Experiment",
                            options=display_names,
                            index=display_names.index(next((display_names[i] for i, name in enumerate(exp_names) if name == st.session_state.selected_file), display_names[0])) if st.session_state.selected_file else 0
                        )
                        
                        # Convert back to original format for file operations
                        selected_idx = display_names.index(selected_display_name)
                        selected_exp_name = exp_names[selected_idx]
                        st.session_state.selected_file = selected_exp_name
                        
                        # Save global state when selection changes
                        self.save_global_state()
                        
                        # Load data
                        selected_exp_path = next(f for f in exp_paths if f.name == selected_exp_name)
                        json_file = selected_exp_path / "steering_data.json"
                        if st.session_state.data is None or st.sidebar.button("Reload Data"):
                            st.session_state.data = self.load_data(json_file)
                            st.session_state.current_exp_path = selected_exp_path
                            
                            # Load steering vectors
                            vectors_file = selected_exp_path / "steering_vectors.pt"
                            st.session_state.steering_vectors = th.load(vectors_file, map_location='cpu')
                        
                        # Load previous state
                        if st.sidebar.button("Load Previous State"):
                            self.load_state(path_info["save_path"])
                        
                        # Save state button
                        if st.sidebar.button("Save Current State"):
                            self.save_state(path_info["save_path"])
    
    def render_generation_comparison(self):
        """Render the generation comparison tab."""
        assert st.session_state.data is not None, "No data loaded"
        
        data = st.session_state.data
        baseline_generations = data.get("baseline_generations", {})
        steered_generations = data.get("steered_generation", [])
        
        if not baseline_generations:
            st.warning("No baseline generations found in the data.")
            return
        
        # Steering vector selection
        st.subheader("Steering Vector Selection")
        num_vectors = len(steered_generations) if steered_generations else 0
        
        if num_vectors > 0:
            vector_options = [f"Steering Vector {i+1}" for i in range(num_vectors)]
            selected_vectors = st.multiselect(
                "Select up to 2 steering vectors",
                options=vector_options,
                default=st.session_state.selected_steering_vectors[:2],
                max_selections=2
            )
            st.session_state.selected_steering_vectors = selected_vectors
            
            # Steering type selection
            st.subheader("Steering Settings")
            steer_type = st.radio(
                "Steering type",
                options=["steer_all", "steer_input"],
                format_func=lambda x: "Steer all tokens" if x == "steer_all" else "Steer input only",
                horizontal=True
            )
            
            # Global steering factor controls
            st.subheader("Global Steering Factors")
            steering_factors_list = data.get("steering_factors", [])
            global_factors = {}
            for vector_name in selected_vectors:
                vector_idx = vector_options.index(vector_name)
                factor = st.selectbox(
                    f"Global factor for {vector_name}",
                    options=steering_factors_list,
                    index=0,
                    key=f"global_factor_{vector_idx}"
                )
                global_factors[vector_name] = factor
            
            # Display generations
            st.subheader("Generation Comparison")
            
            for prompt in baseline_generations.keys():
                st.markdown(f"**Prompt:** {prompt}")
                
                # Create columns for baseline and steered generations
                cols = st.columns([2] + [2] * len(selected_vectors))
                
                # Baseline column
                with cols[0]:
                    st.markdown("**Baseline**")
                    completion = baseline_generations[prompt]
                    self.render_completion(completion, f"baseline_{prompt}", prompt)
                
                # Steered columns
                for i, vector_name in enumerate(selected_vectors):
                    vector_idx = vector_options.index(vector_name)
                    with cols[i + 1]:
                        st.markdown(f"**{vector_name}**")
                        
                        # Get steered generation for this vector
                        if vector_idx < len(steered_generations):
                            completion_key = f"steered_{vector_idx}_{prompt}"
                            steered_data = steered_generations[vector_idx]
                            
                            # Use individual factor if set, otherwise use global factor
                            if completion_key in st.session_state.steering_factors:
                                factor = st.session_state.steering_factors[completion_key]
                            else:
                                factor = global_factors[vector_name]
                            
                            # Use individual steering type if set, otherwise use global type
                            if completion_key in st.session_state.steering_types:
                                completion_steer_type = st.session_state.steering_types[completion_key]
                            else:
                                completion_steer_type = steer_type
                            
                            best_match = None
                            best_diff = float('inf')
                            
                            if prompt in steered_data:
                                prompt_data = steered_data[prompt]
                                for factor_key, generation_data in prompt_data.items():
                                    factor_val = float(factor_key)
                                    diff = abs(factor_val - factor)
                                    if diff < best_diff:
                                        best_diff = diff
                                        best_match = generation_data.get(completion_steer_type, "")
                            
                            if best_match:
                                self.render_completion(best_match, completion_key, prompt, vector_idx, steered_generations)
                            else:
                                st.write("No matching generation found")
                
                st.divider()
    
    def render_completion(self, completion: str, key: str, prompt: str, vector_idx: int = None, steered_generations: list = None):
        """Render a single completion with truncation and expand functionality."""
        if not completion:
            st.write("*No completion available*")
            return
            
        is_expanded = key in st.session_state.expanded_completions
        
        # Native Streamlit approach with dynamic containers
        if len(completion) > MAX_COMPLETION_LENGTH:
            text_container = st.container()
            button_container = st.container()
            
            with text_container:
                if is_expanded:
                    st.text(completion)
                else:
                    truncated = "..." + completion[-MAX_COMPLETION_LENGTH:]
                    st.text(truncated)
            
            with button_container:
                col1, col2 = st.columns([1, 10])
                with col1:
                    if is_expanded:
                        if st.button("▲", key=f"collapse_{key}", help="Collapse"):
                            st.session_state.expanded_completions.discard(key)
                            st.rerun()
                    else:
                        if st.button("▼", key=f"expand_{key}", help="Expand"):
                            st.session_state.expanded_completions.add(key)
                            st.rerun()
        else:
            st.text(completion)
        
        # Individual settings adjustment (only for steered generations)
        if st.session_state.data is not None and key.startswith("steered_"):
            steering_factors_list = st.session_state.data.get("steering_factors", [])
            with st.expander("Adjust Settings"):
                # Steering type selection
                current_steer_type = st.session_state.steering_types.get(key, "steer_all")
                new_steer_type = st.selectbox(
                    "Steering Type",
                    options=["steer_all", "steer_input"],
                    format_func=lambda x: "Steer all tokens" if x == "steer_all" else "Steer input only",
                    index=0 if current_steer_type == "steer_all" else 1,
                    key=f"steer_type_input_{key}"
                )
                
                # Steering factor selection
                current_factor = st.session_state.steering_factors.get(key, steering_factors_list[0] if steering_factors_list else 1.0)
                new_factor = st.selectbox(
                    "Steering Factor",
                    options=steering_factors_list,
                    index=steering_factors_list.index(current_factor) if current_factor in steering_factors_list else 0,
                    key=f"factor_input_{key}"
                )
                
                if st.button("Apply Settings", key=f"apply_{key}"):
                    st.session_state.steering_factors[key] = new_factor
                    st.session_state.steering_types[key] = new_steer_type
                    st.success(f"Settings updated: {new_steer_type}, factor {new_factor}")
                    st.rerun()
    
    def render_pca_visualization(self):
        """Render the PCA visualization tab."""
        if st.session_state.data is None:
            st.warning("Please select a data file from the sidebar.")
            return
        
        if st.session_state.steering_vectors is None:
            st.warning("No steering vectors loaded.")
            return
        
        data = st.session_state.data
        steering_vectors = st.session_state.steering_vectors
        closest_tokens = data.get("closest_tokens", [])
        
        st.subheader("Steering Vectors PCA Visualization")
        
        # Perform PCA on steering vectors
        vectors_np = steering_vectors.detach().cpu().numpy()
        
        if vectors_np.shape[0] < 2:
            st.warning("Need at least 2 steering vectors for PCA visualization.")
            return
        
        pca = PCA(n_components=min(2, vectors_np.shape[0]))
        pca_result = pca.fit_transform(vectors_np)
        
        # Create vector labels with first token
        vector_labels = []
        hover_text = []
        for i, tokens in enumerate(closest_tokens):
            if tokens:
                first_token = list(tokens.keys())[0]
                vector_labels.append(f'{i}: ({first_token})')
                # Create hover text with all tokens
                token_list = ', '.join(list(tokens.keys())[:5])  # Show first 5 tokens
                if len(tokens) > 5:
                    token_list += '...'
                hover_text.append(f'Vector {i}<br>Tokens: {token_list}')
            else:
                vector_labels.append(f'{i}: (no tokens)')
                hover_text.append(f'Vector {i}<br>No tokens')
        
        # Create PCA plot data
        pca_df = pd.DataFrame({
            'PC1': pca_result[:, 0],
            'PC2': pca_result[:, 1] if pca_result.shape[1] > 1 else np.zeros(len(pca_result)),
            'Vector': vector_labels,
            'Hover': hover_text
        })
        
        # Calculate cosine similarity matrix
        cosim_matrix = cosine_similarity(vectors_np)
        
        # Create subplot with PCA on left and heatmap on right
        fig = make_subplots(
            rows=1, cols=2,
            column_widths=[0.6, 0.4],
            subplot_titles=('PCA of Steering Vectors', 'Cosine Similarity Heatmap'),
            specs=[[{"type": "scatter"}, {"type": "heatmap"}]]
        )
        
        # Add PCA scatter plot
        fig.add_trace(
            go.Scatter(
                x=pca_df['PC1'],
                y=pca_df['PC2'],
                mode='markers+text',
                text=pca_df['Vector'],
                hovertext=pca_df['Hover'],
                hoverinfo='text',
                textposition='top center',
                marker=dict(size=10)
            ),
            row=1, col=1
        )
        
        # Add cosine similarity heatmap
        fig.add_trace(
            go.Heatmap(
                z=cosim_matrix,
                x=vector_labels,
                y=vector_labels,
                colorscale='RdBu',
                zmid=0,
                hovertemplate='%{x}<br>%{y}<br>Similarity: %{z:.3f}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # Update layout
        fig.update_xaxes(title_text=f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', row=1, col=1)
        fig.update_yaxes(title_text=f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)' if len(pca.explained_variance_ratio_) > 1 else 'PC2', row=1, col=1)
        fig.update_layout(height=500, showlegend=False)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display closest tokens
        st.subheader("Closest Tokens")
        
        for i, tokens in enumerate(closest_tokens):
            if tokens:
                st.write(f"**Steering Vector {i+1}**")
                
                # Create DataFrame for better display
                token_items = list(tokens.items())[:10]
                df = pd.DataFrame(token_items, columns=['Token', 'Probability'])
                df['Rank'] = range(1, len(df) + 1)
                df = df[['Rank', 'Token', 'Probability']]
                
                # Create plotly table with color-coded rows
                prob_values = df['Probability'].values
                colors = px.colors.sample_colorscale('Blues', prob_values, low=0, high=1)
                
                # Determine text color based on background brightness
                text_colors = ['white' if prob > 0.5 else 'black' for prob in prob_values]
                
                fig = go.Figure(data=[go.Table(
                    header=dict(values=['Rank', 'Token', 'Probability'],
                               fill_color='lightgray',
                               font_color='black',
                               font_size=16,
                               align='left'),
                    cells=dict(values=[df['Rank'], df['Token'], df['Probability'].round(4)],
                              fill_color=[colors, colors, colors],
                              font_color=[text_colors, text_colors, text_colors],
                              font_size=14,
                              align='left'))
                ])
                
                fig.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig, use_container_width=True)
    
    def render_interactive_generation(self):
        """Render the interactive generation tab."""
        if st.session_state.data is None:
            st.warning("Please select a data file from the sidebar.")
            return
        
        if st.session_state.steering_vectors is None:
            st.warning("No steering vectors loaded.")
            return
        
        st.subheader("Interactive Generation")
        
        # Custom prompt input
        custom_prompt = st.text_area(
            "Enter your custom prompt:",
            height=100,
            placeholder="Type your prompt here..."
        )
        
        # Steering vector selection
        data = st.session_state.data
        steering_vectors = st.session_state.steering_vectors
        num_vectors = len(steering_vectors)
        median_norm = data.get("median_norm", 1.0)
        config = data.get("config", {})
        
        if num_vectors > 0:
            vector_options = [f"Steering Vector {i+1}" for i in range(num_vectors)]
            selected_vector = st.selectbox(
                "Select steering vector",
                options=vector_options
            )
            
            # Steering factor controls
            factor_multiplier = st.slider(
                "Steering strength multiplier",
                min_value=0.0,
                max_value=10.0,
                value=1.0,
                step=0.1,
                key="interactive_factor_multiplier"
            )
            
            actual_steering_factor = factor_multiplier * median_norm
            st.write(f"Actual steering factor: {actual_steering_factor:.2f}")
            
            # Steer type selection
            steer_type = st.radio(
                "Steering type",
                options=["all", "input_only"],
                format_func=lambda x: "Steer all tokens" if x == "all" else "Steer input only"
            )
            
            # Generate button
            if st.button("Generate Steered Completions", disabled=not custom_prompt):
                if not custom_prompt:
                    st.warning("Please enter a prompt first.")
                else:
                    # Load model if not loaded or different model
                    model_name = data['model_name']
                    if (st.session_state.loaded_model is None or 
                        getattr(st.session_state.loaded_model, 'model_name', None) != model_name):
                        with st.spinner(f"Loading model {model_name}..."):
                            st.session_state.loaded_model = StandardizedTransformer(
                                model_name, device_map="auto", attn_implementation=None
                            )
                            st.session_state.loaded_model.model_name = model_name
                    
                    vector_idx = vector_options.index(selected_vector)
                    selected_steering_vector = steering_vectors[vector_idx:vector_idx+1]
                    
                    # Create steering model
                    source_layer = config.get("source_layer", 15)
                    target_layer = config.get("target_layer", 15)
                    batch_size = config.get("batch_size", 1)
                    max_new_tokens = config.get("max_new_tokens", 50)
                    do_sample = config.get("do_sample", True)
                    
                    steering_model = SteerSingleModel(
                        st.session_state.loaded_model,
                        source_layer,
                        target_layer,
                        batch_size,
                        steering_factor=median_norm
                    )
                    
                    with st.spinner("Generating completions..."):
                        # Generate baseline
                        with st.session_state.loaded_model.generate(
                            [custom_prompt], max_new_tokens=max_new_tokens, do_sample=do_sample
                        ) as tracer:
                            baseline_output = st.session_state.loaded_model.generator.output.save()
                        baseline_generation = st.session_state.loaded_model.tokenizer.batch_decode(baseline_output)[0]
                        
                        # Generate steered
                        steered_generations = steering_model.steered_generations(
                            [custom_prompt],
                            selected_steering_vector,
                            th.tensor([actual_steering_factor]),
                            steer_type=steer_type,
                            max_new_tokens=max_new_tokens,
                            do_sample=do_sample
                        )
                        steered_generation = steered_generations[0][0][0]
                    
                    # Display results
                    st.subheader("Generated Completions")
                    
                    cols = st.columns(2)
                    
                    with cols[0]:
                        st.markdown("**Baseline**")
                        st.text_area("", value=baseline_generation, height=200, key="baseline_result")
                    
                    with cols[1]:
                        st.markdown(f"**{selected_vector}** (factor: {actual_steering_factor:.2f}, type: {steer_type})")
                        st.text_area("", value=steered_generation, height=200, key="steered_result")
    
    def run(self):
        """Run the Streamlit dashboard."""
        st.set_page_config(
            page_title="Steering Analysis Dashboard",
            page_icon="🎯",
            layout="wide"
        )
        
        # Create sidebar
        self.create_sidebar()
        
        # Main content area with tabs
        tab1, tab2, tab3 = st.tabs(["Generation Comparison", "PCA Visualization", "Interactive Generation"])
        
        with tab1:
            self.render_generation_comparison()
        
        with tab2:
            self.render_pca_visualization()
        
        with tab3:
            self.render_interactive_generation()


if __name__ == "__main__":
    dashboard = SteeringDashboard()
    dashboard.run()