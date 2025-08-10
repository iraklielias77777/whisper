"""
Content Optimization Models for User Whisperer Platform
Thompson Sampling and Neural Contextual Bandits for content optimization
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import random
import logging
import json
import os
from collections import defaultdict, deque
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import joblib

logger = logging.getLogger(__name__)

@dataclass
class Action:
    """Represents a content action/variation"""
    id: str
    features: np.ndarray
    metadata: Dict[str, Any]
    content_type: str = "email"
    template_id: Optional[str] = None
    subject_line: Optional[str] = None
    call_to_action: Optional[str] = None

@dataclass
class Context:
    """Represents the user and environment context"""
    user_features: np.ndarray
    temporal_features: np.ndarray
    historical_features: np.ndarray
    channel_features: np.ndarray
    campaign_features: Optional[np.ndarray] = None

class ThompsonSamplingBandit:
    """
    Thompson Sampling for content optimization with contextual awareness
    """
    
    def __init__(
        self,
        n_actions: int,
        feature_dim: int,
        alpha: float = 1.0,
        beta: float = 1.0,
        decay_factor: float = 0.95,
        context_weight: float = 0.3
    ):
        self.n_actions = n_actions
        self.feature_dim = feature_dim
        self.alpha = alpha
        self.beta = beta
        self.decay_factor = decay_factor
        self.context_weight = context_weight
        
        # Success and failure counts for each action
        self.successes = defaultdict(lambda: alpha)
        self.failures = defaultdict(lambda: beta)
        
        # Contextual parameters
        self.contexts = defaultdict(lambda: deque(maxlen=1000))
        self.rewards = defaultdict(lambda: deque(maxlen=1000))
        self.action_counts = defaultdict(int)
        
        # Time-based decay
        self.last_update = defaultdict(lambda: datetime.now())
        
        logger.info(f"Initialized ThompsonSamplingBandit with {n_actions} actions")
        
    def select_action(
        self,
        available_actions: List[Action],
        context: Context,
        exploration_bonus: float = 0.1
    ) -> Action:
        """Select action using Thompson Sampling with contextual adjustments"""
        
        try:
            if not available_actions:
                raise ValueError("No available actions")
            
            # Sample from Beta distribution for each action
            samples = {}
            
            for action in available_actions:
                # Apply time-based decay
                self._apply_decay(action.id)
                
                # Get success/failure counts
                s = self.successes[action.id]
                f = self.failures[action.id]
                
                # Calculate contextual adjustment
                context_adjustment = self._calculate_context_adjustment(action, context)
                
                # Add exploration bonus for under-explored actions
                exploration_adj = exploration_bonus / (1 + self.action_counts[action.id])
                
                # Adjusted parameters
                adjusted_alpha = s + context_adjustment + exploration_adj
                adjusted_beta = f + (1 - context_adjustment)
                
                # Sample from Beta distribution
                sample = np.random.beta(
                    max(0.01, adjusted_alpha),
                    max(0.01, adjusted_beta)
                )
                
                samples[action.id] = sample
                
                logger.debug(f"Action {action.id}: sample={sample:.3f}, s={s:.2f}, f={f:.2f}, ctx_adj={context_adjustment:.3f}")
            
            # Select action with highest sample
            best_action_id = max(samples, key=samples.get)
            best_action = next(
                a for a in available_actions
                if a.id == best_action_id
            )
            
            # Track selection
            self.action_counts[best_action_id] += 1
            
            logger.info(f"Selected action {best_action_id} with sample {samples[best_action_id]:.3f}")
            
            return best_action
            
        except Exception as e:
            logger.error(f"Action selection failed: {e}")
            # Fallback to random selection
            return random.choice(available_actions)
    
    def _apply_decay(self, action_id: str):
        """Apply time-based decay to action statistics"""
        
        try:
            current_time = datetime.now()
            time_diff = (current_time - self.last_update[action_id]).total_seconds() / 3600  # hours
            
            if time_diff > 0:
                decay = self.decay_factor ** time_diff
                
                # Decay towards prior
                self.successes[action_id] = (
                    self.successes[action_id] * decay + 
                    self.alpha * (1 - decay)
                )
                self.failures[action_id] = (
                    self.failures[action_id] * decay + 
                    self.beta * (1 - decay)
                )
                
                self.last_update[action_id] = current_time
                
        except Exception as e:
            logger.warning(f"Decay application failed for action {action_id}: {e}")
    
    def _calculate_context_adjustment(
        self,
        action: Action,
        context: Context
    ) -> float:
        """Calculate context-based adjustment using similarity weighting"""
        
        try:
            if action.id not in self.contexts or not self.contexts[action.id]:
                return 0.0
            
            # Calculate similarities with historical contexts
            similarities = []
            rewards = list(self.rewards[action.id])
            
            for i, hist_context in enumerate(self.contexts[action.id]):
                if i >= len(rewards):
                    break
                    
                # Multi-dimensional similarity
                user_sim = self._cosine_similarity(
                    context.user_features,
                    hist_context.user_features
                )
                
                temporal_sim = self._cosine_similarity(
                    context.temporal_features,
                    hist_context.temporal_features
                )
                
                # Weighted similarity
                overall_sim = (
                    0.6 * user_sim + 
                    0.2 * temporal_sim + 
                    0.2 * self._cosine_similarity(
                        context.historical_features,
                        hist_context.historical_features
                    )
                )
                
                # Weight by recency
                recency_weight = 0.9 ** (len(similarities))
                weighted_sim = overall_sim * recency_weight
                
                similarities.append((weighted_sim, rewards[i]))
            
            if not similarities:
                return 0.0
            
            # Weighted average of rewards based on similarity
            total_weight = sum(s for s, _ in similarities)
            if total_weight == 0:
                return 0.0
            
            weighted_reward = sum(
                s * r for s, r in similarities
            ) / total_weight
            
            # Apply context weight
            return self.context_weight * weighted_reward
            
        except Exception as e:
            logger.warning(f"Context adjustment calculation failed: {e}")
            return 0.0
    
    def _cosine_similarity(
        self,
        a: np.ndarray,
        b: np.ndarray
    ) -> float:
        """Calculate cosine similarity between two vectors"""
        
        try:
            # Handle different array shapes
            if len(a.shape) > 1:
                a = a.flatten()
            if len(b.shape) > 1:
                b = b.flatten()
            
            # Ensure same length
            min_len = min(len(a), len(b))
            a = a[:min_len]
            b = b[:min_len]
            
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
            
            similarity = dot_product / (norm_a * norm_b)
            return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
            
        except Exception as e:
            logger.warning(f"Cosine similarity calculation failed: {e}")
            return 0.0
    
    def update(
        self,
        action: Action,
        reward: float,
        context: Context,
        reward_type: str = "conversion"
    ):
        """Update model with observed reward"""
        
        try:
            # Normalize reward to [0, 1]
            normalized_reward = max(0.0, min(1.0, reward))
            
            # Update success/failure counts
            if normalized_reward > 0.5:
                self.successes[action.id] += normalized_reward
            else:
                self.failures[action.id] += (1 - normalized_reward)
            
            # Store context and reward for future similarity calculations
            self.contexts[action.id].append(context)
            self.rewards[action.id].append(normalized_reward)
            
            # Update timestamp
            self.last_update[action.id] = datetime.now()
            
            logger.info(f"Updated action {action.id} with reward {normalized_reward:.3f} ({reward_type})")
            
        except Exception as e:
            logger.error(f"Model update failed: {e}")
    
    def get_action_stats(self, action_id: str) -> Dict[str, Any]:
        """Get comprehensive statistics for an action"""
        
        try:
            s = self.successes[action_id]
            f = self.failures[action_id]
            total = s + f
            
            # Calculate confidence interval
            if total > 0:
                mean = s / total
                variance = (s * f) / (total * total * (total + 1))
                std = np.sqrt(variance)
                confidence_95 = 1.96 * std
            else:
                mean = 0.5
                confidence_95 = 0.5
            
            return {
                'action_id': action_id,
                'successes': float(s),
                'failures': float(f),
                'total_trials': float(total),
                'success_rate': float(mean),
                'confidence_interval': {
                    'lower': max(0.0, mean - confidence_95),
                    'upper': min(1.0, mean + confidence_95)
                },
                'confidence_width': float(2 * confidence_95),
                'expected_reward': float(mean),
                'selections': self.action_counts[action_id],
                'last_updated': self.last_update[action_id].isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get stats for action {action_id}: {e}")
            return {
                'action_id': action_id,
                'error': str(e)
            }
    
    def get_all_stats(self) -> Dict[str, Dict]:
        """Get statistics for all actions"""
        
        stats = {}
        for action_id in set(list(self.successes.keys()) + list(self.failures.keys())):
            stats[action_id] = self.get_action_stats(action_id)
        
        return stats
    
    def save_model(self, path: str):
        """Save model state"""
        
        try:
            os.makedirs(path, exist_ok=True)
            
            # Convert defaultdicts and deques to regular dicts/lists for JSON serialization
            model_state = {
                'n_actions': self.n_actions,
                'feature_dim': self.feature_dim,
                'alpha': self.alpha,
                'beta': self.beta,
                'decay_factor': self.decay_factor,
                'context_weight': self.context_weight,
                'successes': dict(self.successes),
                'failures': dict(self.failures),
                'action_counts': dict(self.action_counts),
                'last_update': {
                    k: v.isoformat() for k, v in self.last_update.items()
                },
                'saved_at': datetime.now().isoformat()
            }
            
            # Save contexts and rewards separately (numpy arrays)
            contexts_data = {}
            rewards_data = {}
            
            for action_id in self.contexts:
                contexts_data[action_id] = [
                    {
                        'user_features': ctx.user_features.tolist(),
                        'temporal_features': ctx.temporal_features.tolist(),
                        'historical_features': ctx.historical_features.tolist(),
                        'channel_features': ctx.channel_features.tolist()
                    }
                    for ctx in self.contexts[action_id]
                ]
                rewards_data[action_id] = list(self.rewards[action_id])
            
            # Save main state
            with open(f"{path}/thompson_sampling_state.json", 'w') as f:
                json.dump(model_state, f, indent=2)
            
            # Save contexts and rewards
            with open(f"{path}/thompson_sampling_contexts.json", 'w') as f:
                json.dump(contexts_data, f, indent=2)
            
            with open(f"{path}/thompson_sampling_rewards.json", 'w') as f:
                json.dump(rewards_data, f, indent=2)
            
            logger.info(f"Saved ThompsonSamplingBandit to {path}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
    
    def load_model(self, path: str):
        """Load model state"""
        
        try:
            # Load main state
            with open(f"{path}/thompson_sampling_state.json", 'r') as f:
                model_state = json.load(f)
            
            # Restore parameters
            self.n_actions = model_state['n_actions']
            self.feature_dim = model_state['feature_dim']
            self.alpha = model_state['alpha']
            self.beta = model_state['beta']
            self.decay_factor = model_state['decay_factor']
            self.context_weight = model_state['context_weight']
            
            # Restore statistics
            self.successes = defaultdict(lambda: self.alpha)
            self.successes.update(model_state['successes'])
            
            self.failures = defaultdict(lambda: self.beta)
            self.failures.update(model_state['failures'])
            
            self.action_counts = defaultdict(int)
            self.action_counts.update(model_state['action_counts'])
            
            # Restore timestamps
            self.last_update = defaultdict(lambda: datetime.now())
            for k, v in model_state['last_update'].items():
                self.last_update[k] = datetime.fromisoformat(v)
            
            # Load contexts and rewards
            try:
                with open(f"{path}/thompson_sampling_contexts.json", 'r') as f:
                    contexts_data = json.load(f)
                
                with open(f"{path}/thompson_sampling_rewards.json", 'r') as f:
                    rewards_data = json.load(f)
                
                # Restore contexts
                self.contexts = defaultdict(lambda: deque(maxlen=1000))
                for action_id, ctx_list in contexts_data.items():
                    for ctx_data in ctx_list:
                        context = Context(
                            user_features=np.array(ctx_data['user_features']),
                            temporal_features=np.array(ctx_data['temporal_features']),
                            historical_features=np.array(ctx_data['historical_features']),
                            channel_features=np.array(ctx_data['channel_features'])
                        )
                        self.contexts[action_id].append(context)
                
                # Restore rewards
                self.rewards = defaultdict(lambda: deque(maxlen=1000))
                for action_id, reward_list in rewards_data.items():
                    self.rewards[action_id].extend(reward_list)
                    
            except FileNotFoundError:
                logger.warning("Context/reward files not found, starting with empty history")
            
            logger.info(f"Loaded ThompsonSamplingBandit from {path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise


class NeuralBandit:
    """
    Neural network-based contextual bandit for content optimization
    """
    
    def __init__(
        self,
        feature_dim: int,
        hidden_dims: List[int] = [128, 64, 32],
        learning_rate: float = 0.001,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.01,
        buffer_size: int = 10000,
        batch_size: int = 64
    ):
        self.feature_dim = feature_dim
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        
        # Build neural network
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()
        
        # Experience replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)
        
        # Training tracking
        self.training_step = 0
        self.target_update_frequency = 100
        
        logger.info(f"Initialized NeuralBandit with {feature_dim} features")
        
    def build_model(self) -> keras.Model:
        """Build neural network model for reward prediction"""
        
        try:
            model = models.Sequential(name='neural_bandit')
            
            # Input layer
            model.add(layers.Input(shape=(self.feature_dim,), name='features'))
            
            # Hidden layers with batch normalization and dropout
            for i, dim in enumerate(self.hidden_dims):
                model.add(layers.Dense(
                    dim,
                    activation='relu',
                    kernel_regularizer=keras.regularizers.l2(0.01),
                    name=f'dense_{i+1}'
                ))
                model.add(layers.BatchNormalization(name=f'bn_{i+1}'))
                model.add(layers.Dropout(0.2, name=f'dropout_{i+1}'))
            
            # Output layer (reward prediction)
            model.add(layers.Dense(
                1, 
                activation='sigmoid',
                name='reward_prediction'
            ))
            
            # Compile model
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                loss='binary_crossentropy',
                metrics=['mae', 'mse']
            )
            
            logger.info(f"Built neural bandit model: {model.summary()}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to build neural network: {e}")
            raise
    
    def select_action(
        self,
        available_actions: List[Action],
        context: Context,
        use_uncertainty: bool = True
    ) -> Action:
        """Select action using epsilon-greedy with uncertainty estimation"""
        
        try:
            if not available_actions:
                raise ValueError("No available actions")
            
            # Exploration vs exploitation
            if random.random() < self.epsilon:
                # Exploration: random action
                selected_action = random.choice(available_actions)
                logger.debug(f"Selected random action {selected_action.id} (exploration)")
                return selected_action
            else:
                # Exploitation: best predicted action
                predictions = []
                uncertainties = []
                
                for action in available_actions:
                    features = self._combine_features(action, context)
                    
                    # Get prediction
                    pred = self.model.predict(
                        features.reshape(1, -1),
                        verbose=0
                    )[0][0]
                    
                    # Estimate uncertainty using dropout during inference
                    if use_uncertainty:
                        uncertainty = self._estimate_uncertainty(features, n_samples=10)
                    else:
                        uncertainty = 0.0
                    
                    predictions.append(pred)
                    uncertainties.append(uncertainty)
                
                # Upper confidence bound
                if use_uncertainty:
                    ucb_scores = [
                        pred + 2.0 * unc
                        for pred, unc in zip(predictions, uncertainties)
                    ]
                    best_idx = np.argmax(ucb_scores)
                else:
                    best_idx = np.argmax(predictions)
                
                selected_action = available_actions[best_idx]
                logger.debug(f"Selected action {selected_action.id} with prediction {predictions[best_idx]:.3f}")
                
                return selected_action
                
        except Exception as e:
            logger.error(f"Neural bandit action selection failed: {e}")
            return random.choice(available_actions)
    
    def _combine_features(
        self,
        action: Action,
        context: Context
    ) -> np.ndarray:
        """Combine action and context features"""
        
        try:
            features = []
            
            # Add action features
            action_features = action.features
            if len(action_features.shape) > 1:
                action_features = action_features.flatten()
            features.extend(action_features)
            
            # Add context features
            for context_features in [
                context.user_features,
                context.temporal_features,
                context.historical_features,
                context.channel_features
            ]:
                if context_features is not None:
                    if len(context_features.shape) > 1:
                        context_features = context_features.flatten()
                    features.extend(context_features)
            
            # Add campaign features if available
            if context.campaign_features is not None:
                campaign_features = context.campaign_features
                if len(campaign_features.shape) > 1:
                    campaign_features = campaign_features.flatten()
                features.extend(campaign_features)
            
            # Ensure fixed length
            features = np.array(features)
            if len(features) > self.feature_dim:
                features = features[:self.feature_dim]
            elif len(features) < self.feature_dim:
                # Pad with zeros
                padded = np.zeros(self.feature_dim)
                padded[:len(features)] = features
                features = padded
            
            return features
            
        except Exception as e:
            logger.error(f"Feature combination failed: {e}")
            return np.zeros(self.feature_dim)
    
    def _estimate_uncertainty(
        self,
        features: np.ndarray,
        n_samples: int = 10
    ) -> float:
        """Estimate prediction uncertainty using Monte Carlo Dropout"""
        
        try:
            predictions = []
            
            # Enable dropout during inference for uncertainty estimation
            for _ in range(n_samples):
                # Create a copy of the model with dropout enabled
                pred = self.model(
                    features.reshape(1, -1),
                    training=True  # Enable dropout
                )
                predictions.append(pred.numpy()[0][0])
            
            # Calculate standard deviation as uncertainty measure
            uncertainty = np.std(predictions)
            return uncertainty
            
        except Exception as e:
            logger.warning(f"Uncertainty estimation failed: {e}")
            return 0.0
    
    def update(
        self,
        action: Action,
        reward: float,
        context: Context
    ):
        """Update model with observed reward"""
        
        try:
            # Normalize reward
            normalized_reward = max(0.0, min(1.0, reward))
            
            # Combine features
            features = self._combine_features(action, context)
            
            # Add to replay buffer
            experience = {
                'features': features,
                'reward': normalized_reward,
                'timestamp': datetime.now()
            }
            
            self.replay_buffer.append(experience)
            
            # Train on mini-batch if enough experiences
            if len(self.replay_buffer) >= self.batch_size:
                self._train_on_batch()
            
            logger.debug(f"Updated neural bandit with reward {normalized_reward:.3f}")
            
        except Exception as e:
            logger.error(f"Neural bandit update failed: {e}")
    
    def _train_on_batch(self):
        """Train model on batch from replay buffer"""
        
        try:
            # Sample batch
            batch = random.sample(
                self.replay_buffer,
                min(self.batch_size, len(self.replay_buffer))
            )
            
            # Prepare training data
            X = np.array([exp['features'] for exp in batch])
            y = np.array([exp['reward'] for exp in batch])
            
            # Train
            history = self.model.fit(
                X, y,
                epochs=1,
                verbose=0,
                batch_size=self.batch_size
            )
            
            self.training_step += 1
            
            # Update target model periodically
            if self.training_step % self.target_update_frequency == 0:
                self.update_target_model()
            
            # Decay epsilon
            self.epsilon = max(
                self.min_epsilon,
                self.epsilon * self.epsilon_decay
            )
            
            logger.debug(f"Training step {self.training_step}, loss: {history.history['loss'][0]:.4f}")
            
        except Exception as e:
            logger.error(f"Batch training failed: {e}")
    
    def update_target_model(self):
        """Update target model weights"""
        
        try:
            self.target_model.set_weights(self.model.get_weights())
            logger.debug("Updated target model weights")
            
        except Exception as e:
            logger.warning(f"Target model update failed: {e}")
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get model training statistics"""
        
        return {
            'training_steps': self.training_step,
            'epsilon': self.epsilon,
            'buffer_size': len(self.replay_buffer),
            'model_parameters': self.model.count_params() if self.model else 0,
            'feature_dim': self.feature_dim,
            'hidden_dims': self.hidden_dims
        }
    
    def save_model(self, path: str):
        """Save neural bandit model"""
        
        try:
            os.makedirs(path, exist_ok=True)
            
            # Save Keras models
            self.model.save(f"{path}/neural_bandit_model.h5")
            self.target_model.save(f"{path}/neural_bandit_target.h5")
            
            # Save training state
            state = {
                'feature_dim': self.feature_dim,
                'hidden_dims': self.hidden_dims,
                'learning_rate': self.learning_rate,
                'epsilon': self.epsilon,
                'epsilon_decay': self.epsilon_decay,
                'min_epsilon': self.min_epsilon,
                'buffer_size': self.buffer_size,
                'batch_size': self.batch_size,
                'training_step': self.training_step,
                'target_update_frequency': self.target_update_frequency,
                'saved_at': datetime.now().isoformat()
            }
            
            with open(f"{path}/neural_bandit_state.json", 'w') as f:
                json.dump(state, f, indent=2)
            
            # Save replay buffer (recent experiences only)
            recent_buffer = list(self.replay_buffer)[-1000:]  # Last 1000 experiences
            buffer_data = []
            
            for exp in recent_buffer:
                buffer_data.append({
                    'features': exp['features'].tolist(),
                    'reward': exp['reward'],
                    'timestamp': exp['timestamp'].isoformat()
                })
            
            with open(f"{path}/neural_bandit_buffer.json", 'w') as f:
                json.dump(buffer_data, f, indent=2)
            
            logger.info(f"Saved NeuralBandit to {path}")
            
        except Exception as e:
            logger.error(f"Failed to save neural bandit: {e}")
            raise
    
    def load_model(self, path: str):
        """Load neural bandit model"""
        
        try:
            # Load models
            if os.path.exists(f"{path}/neural_bandit_model.h5"):
                self.model = keras.models.load_model(f"{path}/neural_bandit_model.h5")
            
            if os.path.exists(f"{path}/neural_bandit_target.h5"):
                self.target_model = keras.models.load_model(f"{path}/neural_bandit_target.h5")
            
            # Load state
            if os.path.exists(f"{path}/neural_bandit_state.json"):
                with open(f"{path}/neural_bandit_state.json", 'r') as f:
                    state = json.load(f)
                
                self.feature_dim = state['feature_dim']
                self.hidden_dims = state['hidden_dims']
                self.learning_rate = state['learning_rate']
                self.epsilon = state['epsilon']
                self.epsilon_decay = state['epsilon_decay']
                self.min_epsilon = state['min_epsilon']
                self.buffer_size = state['buffer_size']
                self.batch_size = state['batch_size']
                self.training_step = state['training_step']
                self.target_update_frequency = state['target_update_frequency']
            
            # Load replay buffer
            if os.path.exists(f"{path}/neural_bandit_buffer.json"):
                with open(f"{path}/neural_bandit_buffer.json", 'r') as f:
                    buffer_data = json.load(f)
                
                self.replay_buffer = deque(maxlen=self.buffer_size)
                for exp_data in buffer_data:
                    experience = {
                        'features': np.array(exp_data['features']),
                        'reward': exp_data['reward'],
                        'timestamp': datetime.fromisoformat(exp_data['timestamp'])
                    }
                    self.replay_buffer.append(experience)
            
            logger.info(f"Loaded NeuralBandit from {path}")
            
        except Exception as e:
            logger.error(f"Failed to load neural bandit: {e}")
            raise
