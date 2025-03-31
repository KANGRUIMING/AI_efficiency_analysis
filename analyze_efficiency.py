import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

class EfficiencyAnalyzer:
    """Analyzes videos for task efficiency using a trained model."""
    
    def __init__(self, model_dir='models'):
        """
        Initialize the efficiency analyzer.
        
        Args:
            model_dir: Directory containing the trained model and scaler
        """
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.load_model()
    
    def load_model(self):
        """Load the trained efficiency model and scaler."""
        model_path = os.path.join(self.model_dir, 'efficiency_model.pkl')
        scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            print(f"Error: Model files not found in {self.model_dir}")
            print("Please run train_model.py first to generate the model.")
            return False
        
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            print("Model loaded successfully.")
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def extract_features_from_video(self, video_path, keypoints_dir='keypoints_data', 
                                   visualizations_dir='visualizations', force_extract=False):
        """
        Extract features from a video for efficiency analysis.
        
        Args:
            video_path: Path to the video file
            keypoints_dir: Directory to save keypoints
            visualizations_dir: Directory to save visualizations
            force_extract: Whether to force re-extraction of features
            
        Returns:
            DataFrame with extracted features
        """
        # Ensure directories exist
        Path(keypoints_dir).mkdir(parents=True, exist_ok=True)
        Path(visualizations_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate output CSV path
        video_name = os.path.basename(video_path).split('.')[0]
        output_csv = os.path.join(keypoints_dir, f"{video_name}_keypoints.csv")
        
        # Try to use existing keypoints by default
        if os.path.exists(output_csv) and not force_extract:
            print(f"Using existing keypoints from {output_csv}")
            keypoints_df = pd.read_csv(output_csv)
        # If force_extract is True or the file doesn't exist, extract features
        else:
            if force_extract:
                print(f"Forced re-extraction of features from {video_path}...")
            else:
                print(f"Keypoints file not found. Extracting features from {video_path}...")
            
            keypoints_df = extract_features(
                video_path=video_path, 
                output_csv=output_csv,
                keypoints_dir=keypoints_dir,
                visualizations_dir=visualizations_dir
            )
        
        if keypoints_df is None or len(keypoints_df) == 0:
            print(f"Error: No features extracted from {video_path}")
            return None
        
        # Process the keypoints to extract efficiency features
        from train_model import extract_features_from_keypoints
        features = extract_features_from_keypoints(keypoints_df)
        
        return pd.DataFrame([features])
    
    def analyze_video(self, video_path, keypoints_dir='keypoints_data', 
                     visualizations_dir='visualizations', force_extract=False):
        """
        Analyze a video for task efficiency.
        
        Args:
            video_path: Path to the video file
            keypoints_dir: Directory to save keypoints
            visualizations_dir: Directory to save visualizations
            force_extract: Whether to force re-extraction of features
            
        Returns:
            Dictionary with analysis results
        """
        if self.model is None or self.scaler is None:
            if not self.load_model():
                return {"error": "Model could not be loaded"}
        
        # Extract features
        features_df = self.extract_features_from_video(
            video_path, 
            keypoints_dir, 
            visualizations_dir, 
            force_extract
        )
        
        if features_df is None:
            return {"error": "Feature extraction failed"}
        
        # Scale features
        X_scaled = self.scaler.transform(features_df)
        
        # Make prediction
        efficiency_score = self.model.predict(X_scaled)[0]
        
        # Get feature importance if applicable
        feature_importance = {}
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_names = features_df.columns
            feature_importance = dict(zip(feature_names, importances))
            feature_importance = {k: v for k, v in sorted(
                feature_importance.items(), key=lambda item: item[1], reverse=True
            )}
        
        # Prepare results
        results = {
            "video_path": video_path,
            "efficiency_score": float(efficiency_score),
            "efficiency_percentage": float(efficiency_score * 100),
            "feature_importance": feature_importance,
            "features": features_df.to_dict(orient='records')[0]
        }
        
        # Create efficiency rating
        if efficiency_score >= 0.9:
            results["rating"] = "Excellent"
        elif efficiency_score >= 0.7:
            results["rating"] = "Good"
        elif efficiency_score >= 0.5:
            results["rating"] = "Average"
        elif efficiency_score >= 0.3:
            results["rating"] = "Below Average"
        else:
            results["rating"] = "Poor"
        
        return results
    
    def visualize_results(self, results, output_dir='analysis_results'):
        """
        Create visualizations for the analysis results.
        
        Args:
            results: Analysis results dict
            output_dir: Directory to save visualizations
        """
        if "error" in results:
            print(f"Error: {results['error']}")
            return
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        video_name = os.path.basename(results["video_path"]).split('.')[0]
        
        # Generate visualizations
        
        # 1. Efficiency Score Gauge
        plt.figure(figsize=(10, 6))
        plt.subplot(111, polar=True)
        
        score = results["efficiency_score"]
        theta = np.linspace(0, 180, 100) * np.pi / 180
        r = np.ones_like(theta)
        
        # Create color gradient
        cmap = plt.cm.RdYlGn
        colors = cmap(np.linspace(0, 1, len(theta)))
        
        # Plot gauge background
        plt.bar(theta, r, width=np.pi/50, color=colors, alpha=0.8)
        
        # Plot indicator
        indicator_theta = np.pi - score * np.pi
        plt.plot([indicator_theta, indicator_theta], [0, 1], 'k-', linewidth=4)
        
        # Add labels
        plt.text(np.pi, -0.2, '0%', ha='center', va='center', fontsize=12)
        plt.text(0, -0.2, '100%', ha='center', va='center', fontsize=12)
        plt.text(np.pi/2, -0.2, '50%', ha='center', va='center', fontsize=12)
        
        # Set title and adjust layout
        plt.title(f'Efficiency Score: {results["efficiency_percentage"]:.1f}%\nRating: {results["rating"]}', fontsize=14)
        plt.axis('off')
        
        # Save figure
        plt.savefig(os.path.join(output_dir, f"{video_name}_efficiency_gauge.png"), dpi=300, bbox_inches='tight')
        
        # 2. Feature Importance (if available)
        if results["feature_importance"] and len(results["feature_importance"]) > 0:
            # Get top 10 features
            top_features = dict(list(results["feature_importance"].items())[:10])
            
            plt.figure(figsize=(12, 8))
            plt.barh(list(top_features.keys()), list(top_features.values()))
            plt.xlabel('Importance')
            plt.title('Top Feature Importance')
            plt.gca().invert_yaxis()  # Display from top to bottom
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{video_name}_feature_importance.png"), dpi=300, bbox_inches='tight')
        
        # 3. Generate HTML report
        report_path = os.path.join(output_dir, f"{video_name}_report.html")
        self._generate_html_report(results, report_path)
        
        print(f"Visualizations saved to {output_dir}")
        print(f"Report saved to {report_path}")
    
    def _generate_html_report(self, results, output_path):
        """Generate a detailed HTML report of the efficiency analysis."""
        video_name = os.path.basename(results["video_path"])
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Detailed Efficiency Analysis: {video_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; color: #333; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                .container {{ max-width: 1000px; margin: 0 auto; padding: 20px; background-color: #f9f9f9; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
                .section {{ margin: 30px 0; padding: 20px; background-color: white; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }}
                .score-box {{ 
                    background: linear-gradient(to right, #ff4c4c, #ffeb3b, #4caf50); 
                    height: 30px; 
                    border-radius: 15px; 
                    position: relative; 
                    margin: 20px 0;
                }}
                .score-indicator {{ 
                    position: absolute; 
                    top: -10px; 
                    width: 10px; 
                    height: 50px; 
                    background-color: black; 
                    transform: translateX(-50%);
                }}
                .score-label {{
                    margin-top: 40px;
                    font-size: 18px;
                    font-weight: bold;
                }}
                .metric-container {{
                    display: flex;
                    flex-wrap: wrap;
                    justify-content: space-between;
                }}
                .metric-card {{
                    width: 48%;
                    margin-bottom: 20px;
                    padding: 15px;
                    background-color: #f5f5f5;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                }}
                .metric-name {{
                    font-weight: bold;
                    color: #2c3e50;
                    margin-bottom: 5px;
                }}
                .metric-value {{
                    font-size: 20px;
                    font-weight: bold;
                    color: #3498db;
                }}
                .metric-description {{
                    font-size: 14px;
                    color: #555;
                    margin-top: 5px;
                }}
                .feature-table {{ 
                    width: 100%; 
                    border-collapse: collapse; 
                    margin: 20px 0; 
                }}
                .feature-table th, .feature-table td {{ 
                    border: 1px solid #ddd; 
                    padding: 10px; 
                    text-align: left; 
                }}
                .feature-table th {{ 
                    background-color: #3498db; 
                    color: white;
                }}
                .feature-table tr:nth-child(even) {{
                    background-color: #f2f2f2;
                }}
                .rating {{
                    font-size: 28px;
                    font-weight: bold;
                    margin: 20px 0;
                    text-align: center;
                    padding: 15px;
                    border-radius: 8px;
                }}
                .rating-excellent {{ background-color: #27ae60; color: white; }}
                .rating-good {{ background-color: #2ecc71; color: white; }}
                .rating-average {{ background-color: #f1c40f; color: white; }}
                .rating-below-average {{ background-color: #e67e22; color: white; }}
                .rating-poor {{ background-color: #e74c3c; color: white; }}
                .recommendation {{
                    padding: 10px 15px;
                    margin: 10px 0;
                    border-left: 4px solid #3498db;
                    background-color: #eaf4fd;
                }}
                .insights {{
                    padding: 15px;
                    margin: 15px 0;
                    background-color: #f8f9fa;
                    border-radius: 5px;
                }}
                .insights-title {{
                    font-weight: bold;
                    margin-bottom: 10px;
                    color: #2c3e50;
                }}
                .chart-container {{
                    margin: 30px 0;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Detailed Efficiency Analysis Report</h1>
                <div class="section">
                    <p><strong>Video:</strong> {video_name}</p>
                    <p><strong>Analysis Date:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    
                    <h2>Efficiency Score</h2>
                    <div class="score-box">
                        <div class="score-indicator" style="left: {results['efficiency_percentage']}%;"></div>
                    </div>
                    <div class="score-label">Score: {results['efficiency_percentage']:.1f}%</div>
                    
                    <div class="rating rating-{results['rating'].lower().replace(' ', '-')}">
                        Rating: {results['rating']}
                    </div>
                </div>
                
                <div class="section">
                    <h2>Executive Summary</h2>
        """
        
        # Add executive summary based on the rating
        if results['rating'] == "Excellent":
            html += """
                    <p>The analysis reveals <strong>excellent task efficiency</strong> with highly optimized movements and exceptional coordination.</p>
                    <p>The subject demonstrates expert-level performance with:</p>
                    <ul>
                        <li>Minimal unnecessary movements</li>
                        <li>Optimal hand coordination and synchronization</li>
                        <li>Smooth, purposeful motions with negligible hesitation</li>
                        <li>Excellent spatial awareness and path optimization</li>
                    </ul>
                    <p>This performance sets a high benchmark and can be used as a reference for training.</p>
            """
        elif results['rating'] == "Good":
            html += """
                    <p>The analysis shows <strong>good task efficiency</strong> with well-coordinated movements and effective execution.</p>
                    <p>The subject demonstrates proficient performance with:</p>
                    <ul>
                        <li>Generally smooth and purposeful movements</li>
                        <li>Good bimanual coordination</li>
                        <li>Limited hesitation during task execution</li>
                        <li>Effective path planning</li>
                    </ul>
                    <p>While already performing well, there are minor opportunities for optimization in movement efficiency.</p>
            """
        elif results['rating'] == "Average":
            html += """
                    <p>The analysis indicates <strong>average task efficiency</strong> with adequately coordinated movements.</p>
                    <p>The subject demonstrates standard performance with:</p>
                    <ul>
                        <li>Acceptable but inconsistent movement patterns</li>
                        <li>Moderate bimanual coordination</li>
                        <li>Some noticeable hesitations during execution</li>
                        <li>Room for improvement in path optimization</li>
                    </ul>
                    <p>Targeted practice could yield significant improvements in overall efficiency.</p>
            """
        elif results['rating'] == "Below Average":
            html += """
                    <p>The analysis shows <strong>below-average task efficiency</strong> with suboptimal movement patterns.</p>
                    <p>The subject demonstrates performance that needs improvement in:</p>
                    <ul>
                        <li>Movement smoothness and consistency</li>
                        <li>Hand coordination and synchronization</li>
                        <li>Reducing frequent hesitations</li>
                        <li>Path planning and spatial awareness</li>
                    </ul>
                    <p>A structured training program would be beneficial to address these specific areas.</p>
            """
        else:  # Poor
            html += """
                    <p>The analysis reveals <strong>poor task efficiency</strong> with significant movement inefficiencies.</p>
                    <p>The subject demonstrates performance with substantial limitations in:</p>
                    <ul>
                        <li>Overall movement quality and smoothness</li>
                        <li>Bimanual coordination</li>
                        <li>Task flow with frequent and lengthy hesitations</li>
                        <li>Spatial awareness and movement planning</li>
                    </ul>
                    <p>Comprehensive training focusing on fundamental movement skills is recommended.</p>
            """
        
        html += """
                </div>
                
                <div class="section">
                    <h2>Key Performance Metrics</h2>
                    <div class="metric-container">
        """
        
        # Define descriptions for key metrics
        metric_descriptions = {
            'right_hand_jerk': 'Measures movement smoothness of the right hand. Lower values indicate smoother movements.',
            'left_hand_jerk': 'Measures movement smoothness of the left hand. Lower values indicate smoother movements.',
            'right_hand_total_distance': 'Total distance traveled by the right hand. Efficient movements typically minimize unnecessary distance.',
            'left_hand_total_distance': 'Total distance traveled by the left hand. Efficient movements typically minimize unnecessary distance.',
            'right_hand_avg_speed': 'Average speed of right hand movements. Consistent speeds often indicate deliberate movements.',
            'left_hand_avg_speed': 'Average speed of left hand movements. Consistent speeds often indicate deliberate movements.',
            'avg_hand_separation': 'Average distance between hands during the task. Task-specific optimal separations exist for different activities.',
            'hand_movement_correlation': 'Correlation between left and right hand movements. Higher values indicate better bimanual coordination.',
            'hesitation_count': 'Number of detected hesitations during the task. Fewer hesitations typically indicate better task flow.',
            'right_hand_path_efficiency': 'Ratio of direct path to actual path for the right hand. Higher values (closer to 1.0) indicate more efficient paths.',
            'left_hand_path_efficiency': 'Ratio of direct path to actual path for the left hand. Higher values (closer to 1.0) indicate more efficient paths.'
        }
        
        # Select important metrics to display with detailed descriptions
        key_metrics = [
            'right_hand_jerk', 'left_hand_jerk',
            'right_hand_total_distance', 'left_hand_total_distance',
            'right_hand_avg_speed', 'left_hand_avg_speed',
            'avg_hand_separation', 'hand_movement_correlation',
            'hesitation_count', 'right_hand_path_efficiency',
            'left_hand_path_efficiency'
        ]
        
        # Add key metrics with descriptions and interpretations
        for metric in key_metrics:
            if metric in results["features"]:
                value = results["features"][metric]
                
                # Generate interpretation based on the metric and its value
                interpretation = ""
                if "jerk" in metric and value > 0.5:
                    interpretation = "Higher than optimal. Focus on smoother movements."
                elif "jerk" in metric and value <= 0.5:
                    interpretation = "Good smoothness in movement execution."
                    
                if "correlation" in metric and value < 0.5:
                    interpretation = "Below optimal coordination. Practice synchronized bimanual tasks."
                elif "correlation" in metric and value >= 0.5:
                    interpretation = "Good hand coordination demonstrated."
                    
                if "hesitation" in metric and value > 5:
                    interpretation = "Frequent hesitations detected. Practice for more fluid execution."
                elif "hesitation" in metric and value <= 5:
                    interpretation = "Good flow with minimal hesitations."
                    
                if "path_efficiency" in metric and value < 0.7:
                    interpretation = "Suboptimal path efficiency. Practice more direct movements."
                elif "path_efficiency" in metric and value >= 0.7:
                    interpretation = "Good path optimization demonstrated."
                
                html += f"""
                    <div class="metric-card">
                        <div class="metric-name">{metric.replace('_', ' ').title()}</div>
                        <div class="metric-value">{value:.4f}</div>
                        <div class="metric-description">{metric_descriptions.get(metric, '')}</div>
                        <div class="insights"><strong>Interpretation:</strong> {interpretation}</div>
                    </div>
                """
        
        html += """
                    </div>
                </div>
        """
        
        # Add feature importance if available
        if results["feature_importance"] and len(results["feature_importance"]) > 0:
            html += """
                <div class="section">
                    <h2>Feature Importance Analysis</h2>
                    <p>The following features had the greatest impact on the efficiency score:</p>
                    <table class="feature-table">
                        <tr>
                            <th>Feature</th>
                            <th>Importance</th>
                            <th>Interpretation</th>
                        </tr>
            """
            
            # Add interpretations for top features
            for i, (feature, importance) in enumerate(list(results["feature_importance"].items())[:10]):
                # Generate interpretation for this feature
                interpretation = "This feature significantly influences the efficiency score."
                if "jerk" in feature.lower():
                    interpretation = "Movement smoothness is a key factor in overall efficiency."
                elif "correlation" in feature.lower():
                    interpretation = "Hand coordination strongly impacts task performance."
                elif "hesitation" in feature.lower():
                    interpretation = "Task flow and confidence are important efficiency factors."
                elif "path" in feature.lower():
                    interpretation = "Movement path optimization is crucial for efficient performance."
                elif "speed" in feature.lower():
                    interpretation = "Movement velocity characteristics affect overall efficiency."
                
                html += f"""
                    <tr>
                        <td>{feature.replace('_', ' ').title()}</td>
                        <td>{importance:.4f}</td>
                        <td>{interpretation}</td>
                    </tr>
                """
            
            html += """
                    </table>
                    <p>Understanding these key factors can help focus training and improvement efforts.</p>
                </div>
            """
        
        # Detailed analysis section
        html += """
            <div class="section">
                <h2>Detailed Movement Analysis</h2>
        """
        
        # Hand coordination analysis
        if 'hand_movement_correlation' in results["features"]:
            correlation = results["features"]['hand_movement_correlation']
            if correlation > 0.8:
                html += """
                    <div class="insights">
                        <div class="insights-title">Hand Coordination</div>
                        <p>Excellent bimanual coordination was observed. Both hands worked in highly synchronized patterns, 
                        demonstrating expert-level coordination. This level of synchronization contributes significantly to the overall efficiency.</p>
                    </div>
                """
            elif correlation > 0.6:
                html += """
                    <div class="insights">
                        <div class="insights-title">Hand Coordination</div>
                        <p>Good bimanual coordination was observed. Hands generally worked well together, with occasional 
                        asynchronous movements. Further improvement in hand synchronization could enhance efficiency.</p>
                    </div>
                """
            elif correlation > 0.4:
                html += """
                    <div class="insights">
                        <div class="insights-title">Hand Coordination</div>
                        <p>Moderate bimanual coordination was observed. The hands sometimes worked independently rather than in 
                        complementary patterns. Focused practice on coordinated bimanual tasks is recommended.</p>
                    </div>
                """
            else:
                html += """
                    <div class="insights">
                        <div class="insights-title">Hand Coordination</div>
                        <p>Limited bimanual coordination was observed. The hands frequently moved independently without a clear 
                        coordinated strategy. Significant improvement in hand synchronization is needed.</p>
                    </div>
                """
        
        # Movement smoothness analysis
        if 'right_hand_jerk' in results["features"] and 'left_hand_jerk' in results["features"]:
            right_jerk = results["features"]['right_hand_jerk']
            left_jerk = results["features"]['left_hand_jerk']
            avg_jerk = (right_jerk + left_jerk) / 2
            
            if avg_jerk < 0.3:
                html += """
                    <div class="insights">
                        <div class="insights-title">Movement Smoothness</div>
                        <p>Excellent movement smoothness was observed. Movements were fluid and well-controlled with minimal jerkiness, 
                        indicating high-level motor control and confidence in task execution.</p>
                    </div>
                """
            elif avg_jerk < 0.5:
                html += """
                    <div class="insights">
                        <div class="insights-title">Movement Smoothness</div>
                        <p>Good movement smoothness was observed. Most movements were relatively fluid, with occasional 
                        instances of jerkiness. Refinement of motor control would further enhance performance.</p>
                    </div>
                """
            elif avg_jerk < 0.7:
                html += """
                    <div class="insights">
                        <div class="insights-title">Movement Smoothness</div>
                        <p>Moderate movement smoothness was observed. Movements showed noticeable jerkiness at times, 
                        suggesting opportunities for improved motor control and movement planning.</p>
                    </div>
                """
            else:
                html += """
                    <div class="insights">
                        <div class="insights-title">Movement Smoothness</div>
                        <p>Limited movement smoothness was observed. Movements were frequently jerky and inconsistent, 
                        indicating challenges with motor control. Focused practice on controlled, smooth movements is recommended.</p>
                    </div>
                """
        
        # Hesitation analysis
        if 'hesitation_count' in results["features"]:
            hesitations = results["features"]['hesitation_count']
            
            if hesitations < 3:
                html += """
                    <div class="insights">
                        <div class="insights-title">Task Fluidity</div>
                        <p>Excellent task fluidity was observed with minimal hesitations. The subject moved confidently from 
                        one movement to the next, demonstrating clear task understanding and well-developed motor planning.</p>
                    </div>
                """
            elif hesitations < 6:
                html += """
                    <div class="insights">
                        <div class="insights-title">Task Fluidity</div>
                        <p>Good task fluidity was observed with few hesitations. The subject generally maintained good flow 
                        throughout the task, with occasional brief pauses. Increased familiarity with the task may further reduce hesitations.</p>
                    </div>
                """
            elif hesitations < 10:
                html += """
                    <div class="insights">
                        <div class="insights-title">Task Fluidity</div>
                        <p>Moderate task fluidity with noticeable hesitations. The subject demonstrated intermittent flow disruptions, 
                        suggesting some uncertainty during task execution. Practice to build task confidence is recommended.</p>
                    </div>
                """
            else:
                html += """
                    <div class="insights">
                        <div class="insights-title">Task Fluidity</div>
                        <p>Limited task fluidity with frequent hesitations. The subject showed significant pauses between movements, 
                        indicating uncertainty and potential issues with motor planning. Structured practice focusing on task sequence is highly recommended.</p>
                    </div>
                """
        
        # Path efficiency analysis
        if 'right_hand_path_efficiency' in results["features"] and 'left_hand_path_efficiency' in results["features"]:
            right_path = results["features"]['right_hand_path_efficiency']
            left_path = results["features"]['left_hand_path_efficiency']
            avg_path = (right_path + left_path) / 2
            
            if avg_path > 0.8:
                html += """
                    <div class="insights">
                        <div class="insights-title">Movement Path Optimization</div>
                        <p>Excellent path optimization was observed. Movements followed near-optimal trajectories between targets, 
                        demonstrating superb spatial awareness and movement planning capabilities.</p>
                    </div>
                """
            elif avg_path > 0.7:
                html += """
                    <div class="insights">
                        <div class="insights-title">Movement Path Optimization</div>
                        <p>Good path optimization was observed. Movements generally followed efficient paths, with some minor 
                        deviations from optimal trajectories. Overall spatial awareness is good but could be refined further.</p>
                    </div>
                """
            elif avg_path > 0.6:
                html += """
                    <div class="insights">
                        <div class="insights-title">Movement Path Optimization</div>
                        <p>Moderate path optimization was observed. Movements sometimes took indirect routes between targets, 
                        indicating opportunities for improved spatial planning and awareness.</p>
                    </div>
                """
            else:
                html += """
                    <div class="insights">
                        <div class="insights-title">Movement Path Optimization</div>
                        <p>Limited path optimization was observed. Movements frequently took indirect or inefficient paths, 
                        suggesting challenges with spatial awareness and movement planning. Focused training on direct movement paths is recommended.</p>
                    </div>
                """
        
        html += """
            </div>
            
            <div class="section">
                <h2>Recommendations for Improvement</h2>
                <p>Based on the detailed analysis, the following specific recommendations are provided to improve efficiency:</p>
        """
        
        # Generate specific recommendations based on the metrics
        features = results["features"]
        
        # Targeted recommendations based on detected issues
        recommendations = []
        
        if 'right_hand_jerk' in features and features['right_hand_jerk'] > 0.5:
            recommendations.append("""
                <div class="recommendation">
                    <strong>Improve Right Hand Smoothness:</strong> Practice controlled, slow movements with the right hand, 
                    gradually increasing speed while maintaining smoothness. Focus on eliminating sudden accelerations or decelerations.
                </div>
            """)
            
        if 'left_hand_jerk' in features and features['left_hand_jerk'] > 0.5:
            recommendations.append("""
                <div class="recommendation">
                    <strong>Enhance Left Hand Smoothness:</strong> The left hand shows jerky movements. Practice fluid motion exercises, 
                    such as drawing continuous circles or figure-eights with increasing speed while maintaining smoothness.
                </div>
            """)
            
        if 'hand_movement_correlation' in features and features['hand_movement_correlation'] < 0.6:
            recommendations.append("""
                <div class="recommendation">
                    <strong>Develop Bimanual Coordination:</strong> Practice tasks requiring synchronized hand movements, 
                    such as clapping patterns, bilateral drawing exercises, or simple juggling. Focus on activities where both hands 
                    need to work together in coordinated patterns.
                </div>
            """)
            
        if 'hesitation_count' in features and features['hesitation_count'] > 5:
            recommendations.append("""
                <div class="recommendation">
                    <strong>Reduce Task Hesitations:</strong> Practice the entire task sequence repeatedly to build muscle memory. 
                    Break down complex sequences into smaller segments, master each segment, then gradually combine them. 
                    Visualization exercises can also help improve confidence and reduce hesitations.
                </div>
            """)
            
        if ('right_hand_path_efficiency' in features and features['right_hand_path_efficiency'] < 0.7) or \
           ('left_hand_path_efficiency' in features and features['left_hand_path_efficiency'] < 0.7):
            recommendations.append("""
                <div class="recommendation">
                    <strong>Optimize Movement Paths:</strong> Practice direct point-to-point movements. Use visual guides initially 
                    to train more efficient paths. Focus on conscious planning of movement trajectories before execution, 
                    and develop better spatial awareness through targeted exercises.
                </div>
            """)
        
        # If we have specific speed issues
        if 'right_hand_avg_speed' in features and 'left_hand_avg_speed' in features:
            right_speed = features['right_hand_avg_speed']
            left_speed = features['left_hand_avg_speed']
            speed_diff = abs(right_speed - left_speed)
            
            if speed_diff > 0.1:  # Significant speed difference between hands
                recommendations.append(f"""
                    <div class="recommendation">
                        <strong>Balance Hand Speed:</strong> There is a noticeable speed discrepancy between your hands 
                        (right hand: {right_speed:.3f}, left hand: {left_speed:.3f}). Practice tasks that require both hands 
                        to move at similar speeds. Focus on slowing down the faster hand or speeding up the slower hand to achieve better balance.
                    </div>
                """)
        
        # Add generic recommendations if specific issues weren't detected
        if not recommendations:
            if results['efficiency_percentage'] >= 90:
                recommendations.append("""
                    <div class="recommendation">
                        <strong>Maintain Excellence:</strong> Your performance is already exceptional. Continue practicing to maintain this high level 
                        of efficiency. Consider serving as a model for others or exploring more complex tasks that build on your current skills.
                    </div>
                """)
            else:
                recommendations.append("""
                    <div class="recommendation">
                        <strong>General Improvement:</strong> Regular practice with conscious attention to movement quality will improve overall 
                        efficiency. Record and review your own performances to identify specific areas for improvement.
                    </div>
                """)
                
                recommendations.append("""
                    <div class="recommendation">
                        <strong>Task Familiarization:</strong> Increased familiarity with the task will naturally improve efficiency. 
                        Practice the complete task regularly, focusing on smooth transitions between movement components.
                    </div>
                """)
        
        # Add all recommendations to the report
        for recommendation in recommendations:
            html += recommendation
        
        # Close the report
        html += """
                    <p><em>These recommendations are tailored based on the automated analysis of movement patterns. 
                    Consistent practice focusing on these areas should lead to measurable improvements in task efficiency.</em></p>
                </div>
                
                <div class="section">
                    <h2>Conclusion</h2>
                    <p>This detailed analysis provides an objective assessment of task efficiency based on movement kinematics. 
                    The metrics captured represent various aspects of movement quality that contribute to overall task performance.</p>
                    
                    <p>By addressing the specific recommendations provided, targeted improvements can be made to enhance efficiency. 
                    Periodic reassessment using the same measurement system can track progress over time.</p>
                    
                    <p><em>Analysis completed using automated movement efficiency assessment algorithm.</em></p>
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html)

def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze video for task efficiency")
    parser.add_argument("video_path", help="Path to the video file to analyze")
    parser.add_argument("--model_dir", default="models", help="Directory containing the trained model")
    parser.add_argument("--keypoints_dir", default="C:\\Users\\Ruimi\\OneDrive\\Desktop\\research\\efficiency\\keypoints_data", 
                        help="Directory containing keypoints data")
    parser.add_argument("--visualizations_dir", default="visualizations", help="Directory to save visualization videos")
    parser.add_argument("--results_dir", default="analysis_results", help="Directory to save analysis results")
    parser.add_argument("--force", action="store_true", help="Force re-extraction of features")
    
    args = parser.parse_args()
    
    # Check if video exists
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        return
    
    # Check if model exists
    if not os.path.exists(os.path.join(args.model_dir, 'efficiency_model.pkl')):
        print(f"Error: Model not found in {args.model_dir}")
        print("Please run train_model.py first to train the model.")
        return
    
    # Analyze the video
    analyzer = EfficiencyAnalyzer(model_dir=args.model_dir)
    results = analyzer.analyze_video(
        video_path=args.video_path,
        keypoints_dir=args.keypoints_dir,
        visualizations_dir=args.visualizations_dir,
        force_extract=args.force
    )
    
    if "error" in results:
        print(f"Error: {results['error']}")
        return
    
    # Print results
    print("\n" + "="*50)
    print(f"Efficiency Analysis for {os.path.basename(args.video_path)}")
    print("="*50)
    print(f"Efficiency Score: {results['efficiency_percentage']:.1f}%")
    print(f"Rating: {results['rating']}")
    print("="*50 + "\n")
    
    # Generate visualizations
    analyzer.visualize_results(results, output_dir=args.results_dir)

if __name__ == "__main__":
    # Check if a command-line argument is provided
    import sys
    if len(sys.argv) > 1:
        main()  # Run the original main function with command-line arguments
    else:
        # Use the specific file paths provided
        keypoints_path = r"C:\Users\Ruimi\OneDrive\Desktop\research\efficiency\keypoints_data\IMG_3930_keypoints.csv"
        video_name = os.path.basename(keypoints_path).replace("_keypoints.csv", "")
        
        # Set up directories for output
        model_dir = "models"
        results_dir = "analysis_results"
        
        # Create results directory if it doesn't exist
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        
        # Ensure the keypoints file exists
        if not os.path.exists(keypoints_path):
            print(f"Error: Keypoints file not found at {keypoints_path}")
            sys.exit(1)
        
        print(f"Analyzing keypoints from: {keypoints_path}")
        
        try:
            # Import the necessary modules here to avoid importing MediaPipe
            # The import of extract_features_from_keypoints is moved inside the try block
            # to avoid importing any MediaPipe dependencies until needed
            from train_model import extract_features_from_keypoints
            
            # Load the model
            analyzer = EfficiencyAnalyzer(model_dir=model_dir)
            
            if analyzer.model is None or analyzer.scaler is None:
                print("Error: Model could not be loaded")
                sys.exit(1)
            
            # Load keypoints directly from CSV
            print("Loading keypoints from CSV file...")
            keypoints_df = pd.read_csv(keypoints_path)
            
            # Process the keypoints to extract efficiency features
            features = extract_features_from_keypoints(keypoints_df)
            features_df = pd.DataFrame([features])
            
            # Scale features
            X_scaled = analyzer.scaler.transform(features_df)
            
            # Make prediction
            efficiency_score = analyzer.model.predict(X_scaled)[0]
            
            # Get feature importance if applicable
            feature_importance = {}
            if hasattr(analyzer.model, 'feature_importances_'):
                importances = analyzer.model.feature_importances_
                feature_names = features_df.columns
                feature_importance = dict(zip(feature_names, importances))
                feature_importance = {k: v for k, v in sorted(
                    feature_importance.items(), key=lambda item: item[1], reverse=True
                )}
            
            # Prepare results dictionary
            results = {
                "video_path": f"{video_name}.mp4",  # We don't actually need the video, just a name
                "efficiency_score": float(efficiency_score),
                "efficiency_percentage": float(efficiency_score * 100),
                "feature_importance": feature_importance,
                "features": features_df.to_dict(orient='records')[0]
            }
            
            # Create efficiency rating
            if efficiency_score >= 0.9:
                results["rating"] = "Excellent"
            elif efficiency_score >= 0.7:
                results["rating"] = "Good"
            elif efficiency_score >= 0.5:
                results["rating"] = "Average"
            elif efficiency_score >= 0.3:
                results["rating"] = "Below Average"
            else:
                results["rating"] = "Poor"
            
            # Print results
            print("\n" + "="*50)
            print(f"Efficiency Analysis for {video_name}")
            print("="*50)
            print(f"Efficiency Score: {results['efficiency_percentage']:.1f}%")
            print(f"Rating: {results['rating']}")
            print("="*50 + "\n")
            
            # Generate visualizations
            analyzer.visualize_results(results, output_dir=results_dir)
            
            print(f"Analysis complete. Results saved to {results_dir}")
            
        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            # Check if this is a MediaPipe import error
            if "mediapipe" in str(e).lower():
                print("\nIt seems there's an issue with MediaPipe. Since you're using pre-extracted keypoints,")
                print("let's modify the code to completely avoid MediaPipe:")
                print("\n1. Open analyze_efficiency.py and comment out the line:")
                print("   from extract_features_mediapipe import extract_features")
                print("\n2. In the analyze_video method of EfficiencyAnalyzer class, add a check")
                print("   at the beginning to see if the keypoints file exists, and if so, load it directly")
                print("   instead of calling extract_features()")
            sys.exit(1) 