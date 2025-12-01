import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import networkx as nx

# Page configuration
st.set_page_config(
    page_title="Medical Association Rules",
    page_icon="🔍",
    layout="wide"
)

# Title
st.title("🔍 Medical Association Rule Mining")
st.markdown("Discover hidden patterns in patient data using Apriori algorithm")

# Generate sample medical data with comorbidities
def generate_medical_data():
    """Generate realistic patient data with medical conditions"""
    np.random.seed(42)
    n_patients = 500
    
    data = {
        'Patient_ID': range(1, n_patients + 1),
        'Age': np.random.normal(55, 15, n_patients).astype(int),
        'Gender': np.random.choice(['Male', 'Female'], n_patients),
        'Hypertension': np.random.choice([0, 1], n_patients, p=[0.6, 0.4]),
        'Diabetes': np.random.choice([0, 1], n_patients, p=[0.7, 0.3]),
        'Obesity': np.random.choice([0, 1], n_patients, p=[0.65, 0.35]),
        'Smoker': np.random.choice([0, 1], n_patients, p=[0.75, 0.25]),
        'High_Cholesterol': np.random.choice([0, 1], n_patients, p=[0.6, 0.4]),
        'Heart_Disease': np.random.choice([0, 1], n_patients, p=[0.8, 0.2]),
        'Kidney_Disease': np.random.choice([0, 1], n_patients, p=[0.9, 0.1]),
        'COPD': np.random.choice([0, 1], n_patients, p=[0.85, 0.15]),
    }
    
    df = pd.DataFrame(data)
    
    # Create realistic associations
    # Hypertension + Obesity → Higher chance of Diabetes
    mask = (df['Hypertension'] == 1) & (df['Obesity'] == 1)
    df.loc[mask, 'Diabetes'] = np.random.choice([0, 1], mask.sum(), p=[0.3, 0.7])
    
    # Diabetes + High Cholesterol → Higher chance of Heart Disease
    mask = (df['Diabetes'] == 1) & (df['High_Cholesterol'] == 1)
    df.loc[mask, 'Heart_Disease'] = np.random.choice([0, 1], mask.sum(), p=[0.4, 0.6])
    
    # Smoker → Higher chance of COPD
    mask = (df['Smoker'] == 1)
    df.loc[mask, 'COPD'] = np.random.choice([0, 1], mask.sum(), p=[0.6, 0.4])
    
    return df

# Load data
df = generate_medical_data()

# Sidebar for parameters
st.sidebar.header("⚙️ Algorithm Parameters")

min_support = st.sidebar.slider("Minimum Support", 0.01, 0.5, 0.1, 0.01)
min_confidence = st.sidebar.slider("Minimum Confidence", 0.1, 1.0, 0.5, 0.05)
min_lift = st.sidebar.slider("Minimum Lift", 1.0, 5.0, 1.2, 0.1)

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "🔍 Association Rules", "📈 Visualizations", "💡 Clinical Insights"])

with tab1:
    st.header("📊 Medical Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Patients", len(df))
    with col2:
        st.metric("Conditions Tracked", 8)
    with col3:
        st.metric("Average Age", f"{df['Age'].mean():.1f} years")
    with col4:
        st.metric("Most Common Condition", "Hypertension")
    
    st.subheader("Sample Data")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Condition prevalence
    st.subheader("Condition Prevalence")
    conditions = ['Hypertension', 'Diabetes', 'Obesity', 'Smoker', 
                 'High_Cholesterol', 'Heart_Disease', 'Kidney_Disease', 'COPD']
    
    prevalence = df[conditions].mean().sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    prevalence.plot(kind='bar', ax=ax, color='skyblue')
    ax.set_title('Prevalence of Medical Conditions')
    ax.set_ylabel('Percentage of Patients')
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

with tab2:
    st.header("🔍 Discovered Association Rules")
    
    if st.button("🚀 Mine Association Rules", type="primary"):
        with st.spinner("Mining association rules using Apriori algorithm..."):
            
            # Prepare data for association rules
            medical_conditions = df[conditions]
            
            # Convert to transaction format
            transactions = []
            for _, row in medical_conditions.iterrows():
                transaction = []
                for condition in conditions:
                    if row[condition] == 1:
                        transaction.append(condition)
                transactions.append(transaction)
            
            # Apply TransactionEncoder
            te = TransactionEncoder()
            te_ary = te.fit(transactions).transform(transactions)
            df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
            
            # Apply Apriori algorithm
            frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
            
            if len(frequent_itemsets) > 0:
                # Generate association rules
                rules = association_rules(frequent_itemsets, metric="confidence", 
                                        min_threshold=min_confidence)
                rules = rules[rules['lift'] >= min_lift]
                
                # Sort by confidence and lift
                rules = rules.sort_values(['confidence', 'lift'], ascending=[False, False])
                
                st.success(f"✅ Found {len(rules)} association rules!")
                
                # Display rules in a clean format
                st.subheader("Top Association Rules")
                
                # Create readable rules
                readable_rules = []
                for _, rule in rules.head(20).iterrows():
                    antecedents = list(rule['antecedents'])
                    consequents = list(rule['consequents'])
                    
                    readable_rule = {
                        'Rule': f"{' ∧ '.join(antecedents)} ⇒ {', '.join(consequents)}",
                        'Support': f"{rule['support']:.3f}",
                        'Confidence': f"{rule['confidence']:.3f}",
                        'Lift': f"{rule['lift']:.3f}"
                    }
                    readable_rules.append(readable_rule)
                
                rules_df = pd.DataFrame(readable_rules)
                st.dataframe(rules_df, use_container_width=True)
                
                # Store rules for other tabs
                st.session_state.rules = rules
                st.session_state.readable_rules = readable_rules
                
            else:
                st.warning("No association rules found. Try lowering the minimum support.")

with tab3:
    st.header("📈 Rules Visualization")
    
    if 'rules' in st.session_state and len(st.session_state.rules) > 0:
        rules = st.session_state.rules
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Support vs Confidence")
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(rules['support'], rules['confidence'], 
                               c=rules['lift'], cmap='viridis', alpha=0.6, s=100)
            ax.set_xlabel('Support')
            ax.set_ylabel('Confidence')
            ax.set_title('Support vs Confidence (Color = Lift)')
            plt.colorbar(scatter)
            st.pyplot(fig)
        
        with col2:
            st.subheader("Top Rules by Lift")
            top_rules = rules.nlargest(10, 'lift')
            fig, ax = plt.subplots(figsize=(10, 6))
            y_pos = np.arange(len(top_rules))
            ax.barh(y_pos, top_rules['lift'])
            ax.set_yticks(y_pos)
            
            # Create rule labels
            rule_labels = []
            for _, rule in top_rules.iterrows():
                antecedents = list(rule['antecedents'])
                consequents = list(rule['consequents'])
                rule_labels.append(f"{' + '.join(antecedents[:2])} → {consequents[0]}")
            
            ax.set_yticklabels(rule_labels)
            ax.set_xlabel('Lift')
            ax.set_title('Top Rules by Lift Value')
            st.pyplot(fig)
        
        # Network graph
        st.subheader("Association Network")
        if len(rules) > 0:
            G = nx.DiGraph()
            
            # Add nodes and edges
            for _, rule in rules.head(15).iterrows():
                antecedents = list(rule['antecedents'])
                consequents = list(rule['consequents'])
                
                for ant in antecedents:
                    for cons in consequents:
                        G.add_edge(ant, cons, weight=rule['confidence'])
            
            # Create network plot
            fig, ax = plt.subplots(figsize=(12, 8))
            pos = nx.spring_layout(G, k=1, iterations=50)
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, node_size=1500, node_color='lightblue', 
                                 alpha=0.7, ax=ax)
            
            # Draw edges with width based on confidence
            edge_widths = [G[u][v]['weight'] * 3 for u, v in G.edges()]
            nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6, 
                                 edge_color='gray', arrows=True, ax=ax)
            
            # Draw labels
            nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
            
            ax.set_title('Medical Condition Association Network')
            ax.axis('off')
            st.pyplot(fig)

with tab4:
    st.header("💡 Clinical Insights & Recommendations")
    
    if 'readable_rules' in st.session_state:
        readable_rules = st.session_state.readable_rules
        
        st.info("""
        **Understanding the Metrics:**
        - **Support**: How frequently the rule appears in the dataset
        - **Confidence**: How often the rule is true  
        - **Lift**: How much more likely the consequent is given the antecedent
        """)
        
        st.subheader("Key Clinical Findings")
        
        # Extract top 5 rules for insights
        top_rules = readable_rules[:5]
        
        for i, rule in enumerate(top_rules, 1):
            with st.expander(f"Insight #{i}: {rule['Rule']}"):
                st.write(f"**Support:** {rule['Support']} (occurs in {float(rule['Support'])*100:.1f}% of patients)")
                st.write(f"**Confidence:** {rule['Confidence']} ({float(rule['Confidence'])*100:.1f}% accuracy)")
                st.write(f"**Lift:** {rule['Lift']} ({(float(rule['Lift'])-1)*100:.1f}% more likely than random)")
                
                # Generate clinical recommendation
                rule_parts = rule['Rule'].split(' ⇒ ')
                antecedents = rule_parts[0].split(' ∧ ')
                consequents = rule_parts[1].split(', ')
                
                st.write("**Clinical Recommendation:**")
                st.write(f"Patients with {', '.join(antecedents)} should be screened for {', '.join(consequents)}")
                st.write(f"Consider preventive measures and early intervention strategies")
        
        st.subheader("Preventive Care Strategies")
        st.write("""
        Based on the discovered associations, healthcare providers can:
        
        1. **Implement targeted screening programs** for high-risk patient groups
        2. **Develop personalized prevention plans** based on risk factors
        3. **Optimize resource allocation** for conditions with strong associations
        4. **Educate patients** about risk factor management
        5. **Create clinical decision support** tools using these patterns
        """)

# Footer
st.markdown("---")
st.markdown("🔍 **Medical Association Rule Mining** | Discovering hidden patterns for better healthcare")

print("✅ Association Rules module ready!")