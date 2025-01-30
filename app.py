import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix, 
    mean_squared_error, 
    r2_score
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from xgboost import XGBClassifier, XGBRegressor
import plotly.express as px

class ScreenTimeAnalyzer:
    def __init__(self, df):
        self.df = df
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    def create_visualizations(self):
        st.header("📊 Data Visualization Dashboard")
        
        # Visualization Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Correlation Heatmap", 
            "Distribution Analysis", 
            "Scatter Plots", 
            "Box Plots", 
            "Pair Plot"
        ])
        
        with tab1:
            st.subheader("🔥 Correlation Heatmap")
            plt.figure(figsize=(12, 8))
            try:
                correlation_matrix = self.df[self.numeric_columns].corr()
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                            square=True, linewidths=0.5, cbar_kws={"shrink": .8})
                plt.title("Correlation Heatmap of Numeric Features")
                st.pyplot(plt.gcf())
                plt.close()
            except Exception as e:
                st.error(f"Error creating correlation heatmap: {e}")
        
        with tab2:
            st.subheader("📈 Distribution Analysis")
            dist_column = st.selectbox("Select Column for Distribution", self.numeric_columns)
            
            col1, col2 = st.columns(2)
            
            with col1:
                plt.figure(figsize=(8, 5))
                sns.histplot(self.df[dist_column], kde=True)
                plt.title(f"Histogram of {dist_column}")
                st.pyplot(plt.gcf())
                plt.close()
            
            with col2:
                plt.figure(figsize=(8, 5))
                sns.boxplot(x=self.df[dist_column])
                plt.title(f"Box Plot of {dist_column}")
                st.pyplot(plt.gcf())
                plt.close()
        
        with tab3:
            st.subheader("🔍 Scatter Plot Analysis")
            x_column = st.selectbox("X-axis Column", self.numeric_columns)
            y_column = st.selectbox("Y-axis Column", self.numeric_columns)
            
            fig = px.scatter(
                self.df, 
                x=x_column, 
                y=y_column, 
                title=f"Scatter Plot: {x_column} vs {y_column}",
                labels={x_column: x_column, y_column: y_column},
                hover_data=self.numeric_columns
            )
            st.plotly_chart(fig)
        
        with tab4:
            st.subheader("📦 Box Plots Comparison")
            if self.categorical_columns:
                cat_column = st.selectbox("Select Categorical Column", self.categorical_columns)
                num_column = st.selectbox("Select Numeric Column for Comparison", self.numeric_columns)
                
                plt.figure(figsize=(10, 6))
                sns.boxplot(x=self.df[cat_column], y=self.df[num_column])
                plt.title(f"Box Plot: {num_column} by {cat_column}")
                plt.xticks(rotation=45)
                st.pyplot(plt.gcf())
                plt.close()
            else:
                st.warning("No categorical columns found for box plot comparison")
        
        with tab5:
            st.subheader("🌈 Pair Plot")
            selected_columns = st.multiselect(
                "Select Columns for Pair Plot", 
                self.numeric_columns, 
                default=self.numeric_columns[:min(5, len(self.numeric_columns))]
            )
            
            if len(selected_columns) > 1:
                plt.figure(figsize=(12, 10))
                sns.pairplot(self.df[selected_columns], diag_kind='kde')
                plt.suptitle("Pair Plot of Selected Features", y=1.02)
                st.pyplot(plt.gcf())
                plt.close()
            else:
                st.warning("Select at least two columns for pair plot")
    
    def train_models(self):
        st.header("🤖 Machine Learning Model Analysis")
        
        # Target variable selection
        target_column = st.selectbox("Select target variable", self.numeric_columns)
        
        # Prepare features and target
        X = self.df.drop(columns=[target_column])
        y = self.df[target_column]

        # Ensure all features are numeric
        X = X.select_dtypes(include=[np.number])
        
        # Determine problem type
        is_classification = len(np.unique(y)) <= 10
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Define models based on problem type
        if is_classification:
            models = {
                "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
                "SVM": SVC(probability=True),
                "KNN": KNeighborsClassifier(),
                "Random Forest": RandomForestClassifier(n_estimators=100)
            }
        else:
            models = {
                "XGBoost": XGBRegressor(),
                "SVM": SVR(),
                "KNN": KNeighborsRegressor(),
                "Random Forest": RandomForestRegressor(n_estimators=100)
            }

        # Results storage
        results = {}
        detailed_results = {}
        feature_importance = {}

        # Train and evaluate models
        st.subheader("Model Performance")
        for name, model in models.items():
            st.write(f"Training {name}...")
            try:
                # Fit the model
                model.fit(X_train_scaled, y_train)
                
                # Predict
                y_pred = model.predict(X_test_scaled)
                
                # Evaluate based on problem type
                if is_classification:
                    # Classification metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    results[name] = accuracy
                    detailed_results[name] = classification_report(y_test, y_pred)
                    st.write(f"{name} Accuracy: {accuracy:.2f}")
                    st.text(detailed_results[name])
                    
                    # Confusion Matrix
                    cm = confusion_matrix(y_test, y_pred)
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                    plt.title(f'Confusion Matrix - {name}')
                    st.pyplot(plt.gcf())
                    plt.close()
                else:
                    # Regression metrics
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    results[name] = r2
                    st.write(f"{name} Mean Squared Error: {mse:.2f}")
                    st.write(f"{name} R² Score: {r2:.2f}")
                
                # Feature Importance for tree-based models
                if hasattr(model, 'feature_importances_'):
                    feature_importance[name] = pd.DataFrame({
                        'feature': X.columns,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
                st.write(f"{name} Cross-Validation Scores: {cv_scores}")
                st.write(f"{name} Mean CV Score: {cv_scores.mean():.2f} (±{cv_scores.std()*2:.2f})")
                
            except Exception as e:
                st.error(f"Error training {name}: {e}")

        # Model Comparison and Recommendation
        self.model_recommendation(results, detailed_results, feature_importance)

    def model_recommendation(self, results, detailed_results, feature_importance):
        st.header("🏆 Model Recommendation System")
        
        # Create a comparison table
        comparison_df = pd.DataFrame.from_dict(results, orient='index', columns=['Performance'])
        comparison_df.index.name = 'Model'
        comparison_df = comparison_df.reset_index()
        
        # Display comparison table
        st.dataframe(comparison_df)

        # Determine best model
        best_model = max(results, key=results.get)
        
        # Recommendation Generation
        recommendations = {
            "XGBoost": "Excellent for complex, non-linear relationships. Good for feature importance analysis.",
            "SVM": "Works well with clear margin of separation. Effective for high-dimensional data.",
            "KNN": "Simple, intuitive. Works best with smaller datasets and when decision boundary is irregular.",
            "Random Forest": "Robust to overfitting. Good for feature importance and handling non-linear data."
        }

        st.subheader("🌟 Model Insights and Recommendations")
        
        # Best Model Recommendation
        st.markdown(f"""
        **Best Performing Model:** {best_model}
        **Performance:** {results[best_model]:.4f}
        
        **Recommendation:** {recommendations.get(best_model, "No specific recommendation")}
        """)

        # Detailed Insights
        if best_model in detailed_results:
            st.subheader("Detailed Performance Metrics")
            st.text(detailed_results.get(best_model, "No detailed results available"))

        # Feature Importance
        if best_model in feature_importance:
            st.subheader("Feature Importance")
            st.dataframe(feature_importance[best_model])
            
            # Visualize Feature Importance
            plt.figure(figsize=(10, 6))
            importance_data = feature_importance[best_model]
            sns.barplot(x='importance', y='feature', data=importance_data.head(10))
            plt.title(f'Top 10 Features - {best_model}')
            plt.xlabel('Importance')
            plt.ylabel('Features')
            st.pyplot(plt.gcf())
            plt.close()

        # Contextual Recommendations
        st.subheader("🔍 Contextual Recommendations")
        self.generate_contextual_recommendations()

    def generate_contextual_recommendations(self):
        # Simple recommendation system based on data characteristics
        recommendations = []

        # Screen Time Recommendations
        avg_screen_time = self.df['Screen Time'].mean() if 'Screen Time' in self.df.columns else None
        if avg_screen_time:
            if avg_screen_time > 4:
                recommendations.append("⚠️ High Screen Time Alert: Consider reducing daily screen time")
            elif avg_screen_time < 2:
                recommendations.append("✅ Healthy Screen Time: Maintaining a balanced approach")

        # Health Indicators
        if 'Mental Health Score' in self.df.columns:
            avg_mental_health = self.df['Mental Health Score'].mean()
            if avg_mental_health < 5:
                recommendations.append("🧘 Mental Health Suggestion: Consider stress management techniques")

        # Physical Activity
        if 'Physical Activity' in self.df.columns:
            avg_activity = self.df['Physical Activity'].mean()
            if avg_activity < 30:
                recommendations.append("🏃 Physical Activity Recommendation: Increase daily movement")

        # Display Recommendations
        if recommendations:
            st.markdown("### 💡 Personalized Recommendations")
            for rec in recommendations:
                st.markdown(rec)
        else:
            st.info("No specific recommendations based on current data.")

def main():
    st.title("🖥️ Comprehensive Screen Time Analysis")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            
            # Check if dataframe is empty
            if df.empty:
                st.error("The uploaded CSV file is empty.")
                return
            
            # Create analyzer
            analyzer = ScreenTimeAnalyzer(df)
            
            # Sidebar for navigation
            st.sidebar.title("Navigation")
            analysis_type = st.sidebar.radio("Choose Analysis Type", [
                "Data Visualization", 
                "Machine Learning Analysis", 
                "Comprehensive Report"
            ])
            
            # Conditional Analysis
            if analysis_type == "Data Visualization":
                analyzer.create_visualizations()
            elif analysis_type == "Machine Learning Analysis":
                analyzer.train_models()
            else:
                # Comprehensive Report
                st.header("📄 Comprehensive Analysis Report")
                analyzer.create_visualizations()
                analyzer.train_models()
        
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
