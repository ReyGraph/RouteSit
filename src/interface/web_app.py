import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import json
from typing import Dict, Any, List
import logging
from pathlib import Path

from ..core.reasoning import initialize_routesit_ai
from ..utils.logger import get_logger

logger = get_logger(__name__)

class RoutesitWebApp:
    """Streamlit web application for Routesit AI"""
    
    def __init__(self):
        self.engine = None
        self._initialize()
    
    def _initialize(self):
        """Initialize the Routesit AI engine"""
        try:
            with st.spinner("Initializing Routesit AI..."):
                self.engine = initialize_routesit_ai()
            st.success("Routesit AI initialized successfully!")
        except Exception as e:
            st.error(f"Failed to initialize Routesit AI: {e}")
            self.engine = None
    
    def run(self):
        """Run the Streamlit application"""
        st.set_page_config(
            page_title="Routesit AI - Road Safety Intervention System",
            page_icon="🛣️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Header
        st.title("🛣️ Routesit AI")
        st.subtitle("Road Safety Intervention Decision Intelligence System")
        st.markdown("---")
        
        # Sidebar
        self._render_sidebar()
        
        # Main content
        if self.engine is None:
            st.error("Routesit AI engine failed to initialize. Please check the logs.")
            return
        
        # Main tabs
        tab1, tab2, tab3, tab4 = st.tabs(["🔍 Analysis", "📊 Results", "📈 Comparison", "ℹ️ About"])
        
        with tab1:
            self._render_analysis_tab()
        
        with tab2:
            self._render_results_tab()
        
        with tab3:
            self._render_comparison_tab()
        
        with tab4:
            self._render_about_tab()
    
    def _render_sidebar(self):
        """Render sidebar controls"""
        st.sidebar.header("Configuration")
        
        # Budget constraint
        budget_constraint = st.sidebar.number_input(
            "Budget Constraint (₹)",
            min_value=0,
            max_value=10000000,
            value=500000,
            step=10000,
            help="Maximum budget for interventions"
        )
        
        # Timeline constraint
        timeline_constraint = st.sidebar.number_input(
            "Timeline Constraint (days)",
            min_value=1,
            max_value=365,
            value=30,
            step=1,
            help="Maximum implementation timeline"
        )
        
        # Minimum impact threshold
        min_impact = st.sidebar.slider(
            "Minimum Impact Threshold (%)",
            min_value=0,
            max_value=100,
            value=20,
            step=5,
            help="Minimum required accident reduction percentage"
        )
        
        # Store in session state
        st.session_state.budget_constraint = budget_constraint
        st.session_state.timeline_constraint = timeline_constraint
        st.session_state.min_impact_threshold = min_impact
        
        # System status
        st.sidebar.markdown("---")
        st.sidebar.header("System Status")
        
        if self.engine:
            status = self.engine.get_system_status()
            
            if status['system_status'] == 'operational':
                st.sidebar.success("✅ System Operational")
            else:
                st.sidebar.error("❌ System Error")
            
            # Component status
            for component, status_text in status['components'].items():
                if status_text == 'operational':
                    st.sidebar.success(f"✅ {component.replace('_', ' ').title()}")
                else:
                    st.sidebar.error(f"❌ {component.replace('_', ' ').title()}")
    
    def _render_analysis_tab(self):
        """Render the main analysis tab"""
        st.header("Road Safety Problem Analysis")
        
        # Problem input
        col1, col2 = st.columns([2, 1])
        
        with col1:
            query = st.text_area(
                "Describe the road safety problem:",
                placeholder="e.g., Faded zebra crossing at school zone intersection with high pedestrian traffic",
                height=100,
                help="Provide a detailed description of the road safety issue you want to address"
            )
        
        with col2:
            st.markdown("**Context Information**")
            
            road_type = st.selectbox(
                "Road Type",
                ["Urban", "Rural", "Highway", "Expressway", "Residential", "School Zone", "Hospital Zone"],
                help="Type of road where the problem occurs"
            )
            
            traffic_volume = st.selectbox(
                "Traffic Volume",
                ["Low", "Medium", "High", "Very High"],
                help="Typical traffic volume in the area"
            )
            
            speed_limit = st.number_input(
                "Speed Limit (kmph)",
                min_value=20,
                max_value=120,
                value=50,
                step=10,
                help="Speed limit in the problem area"
            )
        
        # Process button
        if st.button("🔍 Analyze Problem", type="primary"):
            if query.strip():
                self._process_query(query, {
                    'road_type': road_type,
                    'traffic_volume': traffic_volume,
                    'speed_limit': speed_limit
                })
            else:
                st.warning("Please enter a problem description")
    
    def _process_query(self, query: str, context: Dict[str, Any]):
        """Process the user query"""
        try:
            with st.spinner("Analyzing problem and generating recommendations..."):
                # Get constraints from session state
                budget_constraint = st.session_state.get('budget_constraint', 500000)
                timeline_constraint = st.session_state.get('timeline_constraint', 30)
                min_impact_threshold = st.session_state.get('min_impact_threshold', 20)
                
                # Process query
                result = self.engine.process_query(
                    query=query,
                    context=context,
                    budget_constraint=budget_constraint,
                    timeline_constraint=timeline_constraint,
                    min_impact_threshold=min_impact_threshold
                )
                
                # Store result in session state
                st.session_state.analysis_result = result
                
                # Display results
                self._display_analysis_result(result)
                
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            logger.error(f"Query processing failed: {e}")
    
    def _display_analysis_result(self, result: Dict[str, Any]):
        """Display analysis results"""
        st.markdown("---")
        st.header("Analysis Results")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Interventions Found",
                result['analysis']['total_interventions_found']
            )
        
        with col2:
            if result['recommendations']['best_scenario']:
                st.metric(
                    "Estimated Cost",
                    f"₹{result['recommendations']['best_scenario']['total_cost']:,.0f}"
                )
        
        with col3:
            if result['recommendations']['best_scenario']:
                st.metric(
                    "Expected Impact",
                    f"{result['recommendations']['best_scenario']['total_impact']:.1f}%"
                )
        
        with col4:
            if result['recommendations']['best_scenario']:
                st.metric(
                    "Timeline",
                    f"{result['recommendations']['best_scenario']['timeline']} days"
                )
        
        # Reasoning
        if result['analysis'].get('reasoning'):
            st.subheader("AI Reasoning")
            st.write(result['analysis']['reasoning'])
        
        # Validation results
        validation = result['analysis'].get('validation_result', {})
        if validation.get('conflicts') or validation.get('missing_dependencies'):
            st.subheader("⚠️ Validation Issues")
            
            if validation.get('conflicts'):
                st.warning("**Conflicts Detected:**")
                for conflict in validation['conflicts']:
                    st.write(f"- {conflict[0]} conflicts with {conflict[1]}")
            
            if validation.get('missing_dependencies'):
                st.warning("**Missing Dependencies:**")
                for dep in validation['missing_dependencies']:
                    st.write(f"- {dep['intervention']} requires: {', '.join(dep['missing_prerequisites'])}")
        
        # Best recommendation
        if result['recommendations']['best_scenario']:
            st.subheader("🎯 Recommended Solution")
            best_scenario = result['recommendations']['best_scenario']
            
            st.success(f"**Strategy:** {best_scenario['strategy']}")
            st.info(f"**Interventions:** {len(best_scenario['interventions'])} selected")
            
            # Cost-effectiveness
            cost_eff = best_scenario['total_cost'] / max(best_scenario['total_impact'], 1)
            st.metric("Cost-Effectiveness", f"₹{cost_eff:,.0f} per percentage point")
    
    def _render_results_tab(self):
        """Render the detailed results tab"""
        if 'analysis_result' not in st.session_state:
            st.info("Please run an analysis first in the Analysis tab.")
            return
        
        result = st.session_state['analysis_result']
        
        st.header("Detailed Results")
        
        # Optimization scenarios
        if result['recommendations']['optimization_scenarios']:
            st.subheader("Optimization Scenarios")
            
            scenarios_df = pd.DataFrame(result['recommendations']['optimization_scenarios'])
            
            # Display scenarios table
            st.dataframe(
                scenarios_df[['strategy', 'total_cost', 'total_impact', 'cost_effectiveness', 'timeline', 'confidence']],
                use_container_width=True
            )
            
            # Scenario comparison chart
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Cost vs Impact', 'Timeline vs Confidence', 'Cost Distribution', 'Impact Distribution'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Cost vs Impact scatter
            fig.add_trace(
                go.Scatter(
                    x=scenarios_df['total_cost'],
                    y=scenarios_df['total_impact'],
                    mode='markers+text',
                    text=scenarios_df['strategy'],
                    textposition='top center',
                    marker=dict(size=10, color=scenarios_df['confidence'], colorscale='Viridis'),
                    name='Scenarios'
                ),
                row=1, col=1
            )
            
            # Timeline vs Confidence
            fig.add_trace(
                go.Scatter(
                    x=scenarios_df['timeline'],
                    y=scenarios_df['confidence'],
                    mode='markers+text',
                    text=scenarios_df['strategy'],
                    textposition='top center',
                    marker=dict(size=10, color=scenarios_df['cost_effectiveness'], colorscale='Plasma'),
                    name='Timeline vs Confidence'
                ),
                row=1, col=2
            )
            
            # Cost distribution
            fig.add_trace(
                go.Bar(
                    x=scenarios_df['strategy'],
                    y=scenarios_df['total_cost'],
                    name='Cost',
                    marker_color='lightblue'
                ),
                row=2, col=1
            )
            
            # Impact distribution
            fig.add_trace(
                go.Bar(
                    x=scenarios_df['strategy'],
                    y=scenarios_df['total_impact'],
                    name='Impact',
                    marker_color='lightgreen'
                ),
                row=2, col=2
            )
            
            fig.update_layout(height=800, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed interventions
        if result['interventions']['detailed_list']:
            st.subheader("Detailed Interventions")
            
            interventions_df = pd.DataFrame(result['interventions']['detailed_list'])
            
            # Display interventions table
            display_columns = ['intervention_name', 'category', 'problem_type', 'cost_estimate', 'predicted_impact', 'implementation_timeline']
            st.dataframe(
                interventions_df[display_columns],
                use_container_width=True
            )
            
            # Category distribution
            category_counts = interventions_df['category'].value_counts()
            fig_pie = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                title="Interventions by Category"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
    
    def _render_comparison_tab(self):
        """Render the comparison tab"""
        if 'analysis_result' not in st.session_state:
            st.info("Please run an analysis first in the Analysis tab.")
            return
        
        result = st.session_state['analysis_result']
        
        st.header("Scenario Comparison")
        
        if result['recommendations']['scenario_comparison']:
            comparison = result['recommendations']['scenario_comparison']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Best Scenarios")
                
                if 'best_cost_effective' in comparison:
                    st.info(f"**Most Cost-Effective:** {comparison['best_cost_effective']['strategy']}")
                
                if 'highest_impact' in comparison:
                    st.info(f"**Highest Impact:** {comparison['highest_impact']['strategy']}")
                
                if 'fastest_implementation' in comparison:
                    st.info(f"**Fastest Implementation:** {comparison['fastest_implementation']['strategy']}")
            
            with col2:
                st.subheader("Cost Range")
                if 'cost_range' in comparison:
                    cost_range = comparison['cost_range']
                    st.metric("Minimum Cost", f"₹{cost_range['min']:,.0f}")
                    st.metric("Maximum Cost", f"₹{cost_range['max']:,.0f}")
                    st.metric("Average Cost", f"₹{cost_range['avg']:,.0f}")
            
            # Impact range
            if 'impact_range' in comparison:
                st.subheader("Impact Range")
                impact_range = comparison['impact_range']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Minimum Impact", f"{impact_range['min']:.1f}%")
                with col2:
                    st.metric("Maximum Impact", f"{impact_range['max']:.1f}%")
                with col3:
                    st.metric("Average Impact", f"{impact_range['avg']:.1f}%")
    
    def _render_about_tab(self):
        """Render the about tab"""
        st.header("About Routesit AI")
        
        st.markdown("""
        ## 🛣️ Routesit AI - Road Safety Intervention Decision Intelligence System
        
        Routesit AI is an advanced, locally-operable AI system that revolutionizes how road safety interventions are selected, optimized, and implemented. Built for the National Road Safety Hackathon 2025 (IIT Madras), this system provides comprehensive decision intelligence for road safety improvements.
        
        ### Key Features
        
        - **Multi-modal Input Processing**: Text + Images + Metadata fusion
        - **Intelligent Intervention Matching**: 300+ interventions with semantic search
        - **Dependency Analysis**: Conflict detection and prerequisite checking
        - **Cost-Benefit Optimization**: Quantitative impact modeling
        - **Scenario Comparison**: Multi-objective optimization engine
        - **Implementation Planning**: Ready-to-deploy action items
        - **Local Processing**: No cloud dependencies, runs entirely offline
        
        ### Technology Stack
        
        - **Local LLM**: Llama 2 7B (quantized) for reasoning
        - **Vector Search**: ChromaDB with sentence-transformers
        - **Graph Processing**: NetworkX for dependency modeling
        - **Optimization**: SciPy for multi-objective optimization
        - **Computer Vision**: YOLOv8 for road sign detection
        - **Web Interface**: Streamlit for user interaction
        
        ### How It Works
        
        1. **Input Processing**: Analyze problem description and context
        2. **Intervention Retrieval**: Find relevant interventions using semantic search
        3. **Dependency Validation**: Check for conflicts and prerequisites
        4. **Optimization**: Generate multiple scenarios using different strategies
        5. **Reasoning**: Generate AI-powered explanations and recommendations
        6. **Output**: Provide actionable implementation plans
        
        ### Database
        
        The system includes a comprehensive database of 300+ road safety interventions covering:
        
        - **Road Signs**: Regulatory, warning, and information signs
        - **Road Markings**: Lane markings, crossings, and directional indicators
        - **Traffic Calming**: Speed humps, chicanes, and traffic circles
        - **Infrastructure**: Lighting, barriers, and pedestrian facilities
        
        Each intervention includes:
        - Cost estimates (materials + labor)
        - Predicted safety impact
        - Implementation timeline
        - Compliance requirements
        - Dependencies and conflicts
        
        ### Compliance Standards
        
        All recommendations comply with:
        - IRC 67-2022 (Road Signs)
        - IRC 35-2015 (Road Markings)
        - IRC 103-2012 (Traffic Calming)
        - MoRTH Guidelines
        - Local Authority Requirements
        
        ### Development Team
        
        Built for the National Road Safety Hackathon 2025 at IIT Madras, demonstrating:
        - Technical innovation in AI and optimization
        - Practical application to real-world road safety challenges
        - Complete local implementation without cloud dependencies
        - Comprehensive decision intelligence capabilities
        
        ### Contact
        
        For questions or support, please refer to the hackathon documentation or contact the development team.
        """)
        
        # System statistics
        if self.engine:
            st.subheader("System Statistics")
            status = self.engine.get_system_status()
            
            if 'statistics' in status:
                stats = status['statistics']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'vector_search' in stats:
                        vs_stats = stats['vector_search']
                        st.metric("Total Interventions", vs_stats.get('total_interventions', 0))
                        st.metric("Embedding Model", vs_stats.get('embedding_model', 'N/A'))
                
                with col2:
                    if 'dependency_graph' in stats:
                        dg_stats = stats['dependency_graph']
                        st.metric("Dependencies", dg_stats.get('total_dependencies', 0))
                        st.metric("Conflicts", dg_stats.get('total_conflicts', 0))
                        st.metric("Synergies", dg_stats.get('total_synergies', 0))

def main():
    """Main function to run the Streamlit app"""
    app = RoutesitWebApp()
    app.run()

if __name__ == "__main__":
    main()
