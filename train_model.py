import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge

def load_and_process_keypoints(keypoints_dir, perfect_video_names):
    """Load keypoints from perfect videos and prepare for training."""
    all_data = []
    perfect_count = 0
    nonperfect_count = 0
    
    for filename in os.listdir(keypoints_dir):
        if filename.endswith('_keypoints.csv'):
            # Check if this is from a perfect video
            video_name = filename.replace('_keypoints.csv', '')
            is_perfect = any(perfect_name in video_name for perfect_name in perfect_video_names)
            
            # Load data
            data_path = os.path.join(keypoints_dir, filename)
            df = pd.read_csv(data_path)
            
            # Extract features
            features = extract_features_from_keypoints(df)
            features['is_perfect'] = 1 if is_perfect else 0
            
            # Count samples by class
            if is_perfect:
                perfect_count += 1
                
                # Apply data augmentation to perfect videos to increase samples
                if len(df) > 10:  # Only augment if we have enough frames
                    # Create 3 augmented versions of each perfect video
                    for i in range(3):
                        augmented_features = extract_features_from_keypoints(
                            augment_single_keypoints_df(df, seed=i)
                        )
                        augmented_features['is_perfect'] = 1
                        all_data.append(augmented_features)
                        perfect_count += 1
            else:
                nonperfect_count += 1
            
            all_data.append(features)
    
    print(f"Dataset statistics:")
    print(f"  - Perfect videos: {perfect_count}")
    print(f"  - Non-perfect videos: {nonperfect_count}")
    print(f"  - Total samples: {len(all_data)}")
    
    return pd.DataFrame(all_data)

def extract_features_from_keypoints(df):
    """Extract meaningful features from raw keypoints data."""
    features = {}
    
    # First, determine if this is MediaPipe data or simple motion tracking data
    if 'pose_0_x' in df.columns:  # This is MediaPipe data
        # 1. Movement smoothness - calculate jerk (derivative of acceleration)
        if 'pose_16_x' in df.columns:  # Right wrist
            x = df['pose_16_x'].values
            y = df['pose_16_y'].values
            
            # Calculate velocities (first differences)
            vx = np.diff(x)
            vy = np.diff(y)
            
            # Calculate accelerations (second differences)
            ax = np.diff(vx)
            ay = np.diff(vy)
            
            # Calculate jerk (third differences)
            jx = np.diff(ax)
            jy = np.diff(ay)
            
            # Total jerk
            total_jerk = np.sum(np.sqrt(jx**2 + jy**2))
            features['right_hand_jerk'] = total_jerk
            
            # Total distance traveled
            distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
            features['right_hand_total_distance'] = np.sum(distances)
            
            # Average speed
            if 'timestamp' in df.columns:
                time_diff = np.diff(df['timestamp'].values)
                speeds = distances / time_diff
                features['right_hand_avg_speed'] = np.mean(speeds)
                features['right_hand_max_speed'] = np.max(speeds)
            
        # Left wrist analysis (pose_15)
        if 'pose_15_x' in df.columns:
            x = df['pose_15_x'].values
            y = df['pose_15_y'].values
            
            # Calculate velocities and distances for left hand
            distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
            features['left_hand_total_distance'] = np.sum(distances)
            
            vx = np.diff(x)
            vy = np.diff(y)
            ax = np.diff(vx)
            ay = np.diff(vy)
            jx = np.diff(ax)
            jy = np.diff(ay)
            features['left_hand_jerk'] = np.sum(np.sqrt(jx**2 + jy**2))
            
            if 'timestamp' in df.columns:
                time_diff = np.diff(df['timestamp'].values)
                speeds = distances / time_diff
                features['left_hand_avg_speed'] = np.mean(speeds)
                features['left_hand_max_speed'] = np.max(speeds)
        
        # 2. Bimanual coordination (how well both hands work together)
        if 'pose_15_x' in df.columns and 'pose_16_x' in df.columns:
            # Calculate distance between hands over time
            hand_distances = np.sqrt(
                (df['pose_15_x'] - df['pose_16_x'])**2 + 
                (df['pose_15_y'] - df['pose_16_y'])**2
            )
            features['avg_hand_separation'] = np.mean(hand_distances)
            features['std_hand_separation'] = np.std(hand_distances)
            
            # Correlation between hand movements
            if len(df) > 5:  # Need enough frames
                right_movement = np.sqrt(np.diff(df['pose_16_x'])**2 + np.diff(df['pose_16_y'])**2)
                left_movement = np.sqrt(np.diff(df['pose_15_x'])**2 + np.diff(df['pose_15_y'])**2)
                if len(right_movement) > 1 and len(left_movement) > 1:
                    correlation = np.corrcoef(right_movement, left_movement)[0, 1]
                    features['hand_movement_correlation'] = correlation if not np.isnan(correlation) else 0
        
        # 3. Posture stability
        if 'pose_11_x' in df.columns and 'pose_12_x' in df.columns:
            # Shoulders stability (11, 12 are shoulders in MediaPipe)
            shoulder_x = (df['pose_11_x'] + df['pose_12_x']) / 2
            shoulder_y = (df['pose_11_y'] + df['pose_12_y']) / 2
            features['shoulder_x_stability'] = np.std(shoulder_x)
            features['shoulder_y_stability'] = np.std(shoulder_y)
        
        # 4. Temporal efficiency
        if 'timestamp' in df.columns:
            features['total_time'] = df['timestamp'].max() - df['timestamp'].min()
        features['total_frames'] = len(df)
        
        # 5. Hand trajectory efficiency (straightness of movement)
        if 'pose_16_x' in df.columns:
            # Right hand
            points = np.column_stack([df['pose_16_x'], df['pose_16_y']])
            
            # Split into movement segments where velocity changes significantly
            segments = []
            segment = [points[0]]
            for i in range(1, len(points)):
                segment.append(points[i])
                if i < len(points) - 1:
                    # Calculate angle change in trajectory
                    if i > 0 and i < len(points) - 1:
                        v1 = points[i] - points[i-1]
                        v2 = points[i+1] - points[i]
                        if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                            dot_product = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                            angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
                            # If angle is large, start new segment
                            if angle > 0.5:  # ~30 degrees
                                segments.append(np.array(segment))
                                segment = [points[i]]
            
            if segment:
                segments.append(np.array(segment))
            
            # Calculate straightness of each segment (ratio of direct distance to actual path)
            straightness_values = []
            for segment in segments:
                if len(segment) > 2:
                    direct_distance = np.linalg.norm(segment[-1] - segment[0])
                    path_distance = np.sum(np.linalg.norm(segment[1:] - segment[:-1], axis=1))
                    if path_distance > 0:
                        straightness = direct_distance / path_distance
                        straightness_values.append(straightness)
            
            if straightness_values:
                features['right_hand_path_efficiency'] = np.mean(straightness_values)
        
        # 6. Hesitation detection (pauses in movement)
        if 'pose_16_x' in df.columns and 'timestamp' in df.columns:
            # Calculate instantaneous speed
            dx = np.diff(df['pose_16_x'])
            dy = np.diff(df['pose_16_y'])
            dt = np.diff(df['timestamp'])
            speeds = np.sqrt(dx**2 + dy**2) / dt
            
            # Count moments of very low speed (hesitations)
            low_speed_threshold = np.percentile(speeds, 15)  # Bottom 15% of speeds
            hesitation_mask = speeds < low_speed_threshold
            hesitation_count = np.sum(hesitation_mask)
            
            features['hesitation_count'] = hesitation_count
            features['hesitation_ratio'] = hesitation_count / len(speeds) if len(speeds) > 0 else 0
        
        # 7. Hand precision (for assembly tasks)
        if 'hand_0_0_x' in df.columns:
            # Calculate fingertip stability for precision tasks
            for finger_id in [4, 8, 12, 16, 20]:  # Fingertips in MediaPipe hand model
                if f'hand_0_{finger_id}_x' in df.columns:
                    x = df[f'hand_0_{finger_id}_x'].values
                    y = df[f'hand_0_{finger_id}_y'].values
                    
                    # Measure stability when fingertip is slow-moving (likely during precision tasks)
                    if len(x) > 2:
                        velocities = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
                        low_velocity_mask = velocities < np.percentile(velocities, 25)
                        
                        if np.sum(low_velocity_mask) > 0:
                            # Calculate position variation during slow movements
                            x_stability = np.std(x[1:][low_velocity_mask])
                            y_stability = np.std(y[1:][low_velocity_mask])
                            features[f'finger_{finger_id}_precision'] = x_stability + y_stability
        
    elif 'avg_motion' in df.columns:  # This is simple motion tracking data
        # Extract features from simple motion tracking data
        features['total_frames'] = len(df)
        
        if 'timestamp' in df.columns:
            features['total_time'] = df['timestamp'].max() - df['timestamp'].min()
        
        if 'avg_motion' in df.columns:
            features['avg_motion'] = df['avg_motion'].mean()
            features['max_motion'] = df['avg_motion'].max()
            features['motion_std'] = df['avg_motion'].std()
            
            # Motion consistency (coefficient of variation)
            mean_motion = df['avg_motion'].mean()
            std_motion = df['avg_motion'].std()
            if mean_motion > 0:
                features['motion_cv'] = std_motion / mean_motion
            
            # Detect phases of movement vs. stillness
            motion_threshold = np.percentile(df['avg_motion'].values, 25)
            is_moving = df['avg_motion'] > motion_threshold
            
            # Count transitions between moving and still
            transitions = np.sum(np.abs(np.diff(is_moving.astype(int))))
            features['motion_phase_changes'] = transitions
        
        if 'motion_variance' in df.columns:
            features['avg_jerk'] = df['motion_variance'].mean()
            features['max_jerk'] = df['motion_variance'].max()
        
        if 'num_points' in df.columns:
            features['avg_tracked_points'] = df['num_points'].mean()
            
            # Stability of tracking (higher is better)
            features['tracking_stability'] = 1.0 - (df['num_points'].std() / (df['num_points'].mean() if df['num_points'].mean() > 0 else 1))
        
        if 'avg_angle' in df.columns:
            # Analyze directional consistency
            angles = df['avg_angle'].values
            angle_diffs = np.abs(np.diff(angles))
            # Adjust differences greater than pi
            angle_diffs = np.minimum(angle_diffs, 2*np.pi - angle_diffs)
            features['direction_consistency'] = 1.0 - (np.mean(angle_diffs) / np.pi)
    
    return features

def augment_single_keypoints_df(df, seed=42):
    """Create a single augmented version of keypoint data."""
    np.random.seed(seed)
    
    # Create copy of dataframe
    augmented_df = df.copy()
    
    # Add slight noise to positions (jitter)
    pose_columns = [col for col in augmented_df.columns if col.startswith('pose_') and ('_x' in col or '_y' in col)]
    for col in pose_columns:
        # Add small random noise
        noise = np.random.normal(0, 0.01, len(augmented_df))
        augmented_df[col] = augmented_df[col] + noise
    
    # Time scaling (slightly faster or slower)
    if 'timestamp' in augmented_df.columns:
        # Random scaling factor between 0.9 and 1.1
        scale_factor = 0.9 + 0.2 * np.random.random()
        augmented_df['timestamp'] = augmented_df['timestamp'] * scale_factor
    
    return augmented_df

def train_efficiency_model(data):
    """Train a model for video efficiency regression (continuous score instead of binary)."""
    # Check if we have enough data
    if len(data) < 6:
        print("WARNING: Not enough data for reliable model training.")
        print("Please add more videos to your dataset.")
        return None, None
    
    # Prepare data
    X = data.drop(columns=['is_perfect'])
    
    # Convert binary labels to continuous scores (0.0 to 1.0)
    # If we already have continuous scores, this preserves them
    if set(data['is_perfect'].unique()) == {0, 1}:
        print("Converting binary labels to continuous efficiency scores")
        # Perfect videos get score 0.9-1.0, imperfect get 0.0-0.5
        y = data['is_perfect'].apply(lambda x: np.random.uniform(0.9, 1.0) if x == 1 else np.random.uniform(0.0, 0.5))
    else:
        y = data['is_perfect']  # Already continuous
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # If we have very few samples, skip cross-validation
    if len(X) < 10:
        print("Dataset too small for cross-validation, using 70/30 train/test split instead.")
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42
        )
        
        # Try different regression models
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=20, max_depth=3, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, random_state=42),
            'Ridge': Ridge(alpha=1.0),
            'LinearRegression': LinearRegression()
        }
        
        best_model = None
        best_score = -float('inf')
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            print(f"{name} - Train R²: {train_score:.4f}, Test R²: {test_score:.4f}")
            
            if test_score > best_score:
                best_score = test_score
                best_model = model
        
        # Train final model on all data
        best_model.fit(X_scaled, y)
    else:
        # We have enough data for cross-validation
        print("Using 5-fold cross-validation for model evaluation")
        model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        
        try:
            cv_scores = cross_val_score(model, X_scaled, y, cv=5)
            print(f"Cross-validation R² scores: {cv_scores}")
            print(f"Mean CV R² score: {np.mean(cv_scores):.4f}")
        except Exception as e:
            print(f"Cross-validation failed: {str(e)}")
            print("Falling back to regular training...")
        
        # Train final model on all data
        model.fit(X_scaled, y)
        best_model = model
    
    # Save model and scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/efficiency_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    
    return best_model, scaler

if __name__ == "__main__":
    # List of perfect video names (without extension)
    perfect_video_names = ["IMG_3928", "IMG_3929", "IMG_3930"]  # Your perfect videos
    
    # Process data and train model
    print("Loading and processing keypoints data...")
    data = load_and_process_keypoints("keypoints_data", perfect_video_names)
    
    if len(data) > 0:
        print("Training efficiency model...")
        model, scaler = train_efficiency_model(data)
        if model is not None:
            print("Model training completed successfully!")
    else:
        print("No data found. Please run extract_features.py first.") 