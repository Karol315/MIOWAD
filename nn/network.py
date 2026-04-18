import numpy as np
from tqdm.notebook import tqdm
from .layer import Layer
from .activation_functions import Softmax, Sigmoid, ReLU, Linear, Tanh


class Network:
    def __init__(self, layers, task="classification"):
        self.layers = layers
        self.task = task
        self.pbar = None
        self._es_state = None

        last_activation = self.layers[-1].activation

        if self.task == "classification":
            if not isinstance(last_activation, (Softmax, Sigmoid)):
                raise ValueError("W trybie klasyfikacji ostatnia warstwa musi używać funkcji Softmax lub Sigmoid.")
        elif self.task == "regression":
            if not isinstance(last_activation, (Linear, Sigmoid, ReLU, Tanh)):
                raise ValueError("W trybie regresji ostatnia warstwa musi używać funkcji Linear, Sigmoid lub ReLU.")
        else:
            raise ValueError("Nieznany parametr 'task'. Użyj 'classification' lub 'regression'.")

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        current_output = np.asarray(inputs)
        for layer in self.layers:
            current_output = layer.forward(current_output)
        return current_output

    def fit(self, X, y, epochs, learning_rate, batch_size=None, mode="simple", momentum=0.9, decay_rate=0.9,
            l2_lambda=0.0, validation_data=None, patience=None, tqdm_desc="Trening", total_epochs=None):
        n_samples = X.shape[0]
        if batch_size is None:
            batch_size = n_samples

        actual_total = total_epochs if total_epochs is not None else epochs

        if isinstance(patience, float):
            patience = max(1, int(patience * actual_total))

        if not hasattr(self, 'pbar') or self.pbar is None:
            self.pbar = tqdm(total=actual_total, desc=tqdm_desc, leave=True)
            self._es_state = {
                'best_val_loss': float('inf'),
                'patience_counter': 0,
                'best_weights': None,
                'best_biases': None
            }

        for epoch in range(epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                output = self.predict(X_batch)

                if self.task == "classification":
                    d_output = (output - y_batch) / len(X_batch)
                else:
                    d_output = 2 * (output - y_batch) / len(X_batch)

                for layer in reversed(self.layers):
                    d_output = layer.backward(d_output, learning_rate, mode=mode, momentum=momentum,
                                              decay_rate=decay_rate, l2_lambda=l2_lambda)

            self.pbar.update(1)

            if validation_data is not None:
                X_val, y_val = validation_data
                val_output = self.predict(X_val)

                if self.task == "classification":
                    val_loss = -np.sum(y_val * np.log(val_output + 1e-9)) / len(X_val)
                else:
                    val_loss = np.mean((y_val - val_output) ** 2)

                self.pbar.set_postfix({'val_loss': f"{val_loss:.6f}"})

                if patience is not None:
                    if val_loss < self._es_state['best_val_loss']:
                        self._es_state['best_val_loss'] = val_loss
                        self._es_state['patience_counter'] = 0
                        self._es_state['best_weights'] = [layer.weights.copy() for layer in self.layers]
                        self._es_state['best_biases'] = [layer.biases.copy() for layer in self.layers]
                    else:
                        self._es_state['patience_counter'] += 1
                        if self._es_state['patience_counter'] >= patience:
                            tqdm.write(
                                f"\n[Early Stopping] Aktywowano! Najlepszy błąd walidacji: {self._es_state['best_val_loss']:.6f}")
                            for idx, layer in enumerate(self.layers):
                                layer.weights = self._es_state['best_weights'][idx]
                                layer.biases = self._es_state['best_biases'][idx]

                            self.pbar.close()
                            self.pbar = None
                            self._es_state = None
                            break

        if hasattr(self, 'pbar') and self.pbar is not None and self.pbar.n >= self.pbar.total:
            self.pbar.close()
            self.pbar = None
            self._es_state = None

    def fit_with_history(self, X, y, epochs, learning_rate, batch_size=None, mode="simple", momentum=0.9,
                         decay_rate=0.9, l2_lambda=0.0, validation_data=None, patience=None, tqdm_desc="Trening",
                         total_epochs=None):
        n_samples = X.shape[0]
        if batch_size is None:
            batch_size = n_samples

        history = {
            'loss': [],
            'val_loss': [],
            'weights': {i: [] for i in range(len(self.layers))}
        }

        actual_total = total_epochs if total_epochs is not None else epochs

        if isinstance(patience, float):
            patience = max(1, int(patience * actual_total))

        if not hasattr(self, 'pbar') or self.pbar is None:
            self.pbar = tqdm(total=actual_total, desc=tqdm_desc, leave=True)
            self._es_state = {
                'best_val_loss': float('inf'),
                'patience_counter': 0,
                'best_weights': None,
                'best_biases': None
            }

        for epoch in range(epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0

            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                output = self.predict(X_batch)

                if self.task == "classification":
                    loss = -np.sum(y_batch * np.log(output + 1e-9)) / len(X_batch)
                    d_output = (output - y_batch) / len(X_batch)
                else:
                    loss = np.mean((y_batch - output) ** 2)
                    d_output = 2 * (output - y_batch) / len(X_batch)

                epoch_loss += loss * len(X_batch)

                for layer in reversed(self.layers):
                    d_output = layer.backward(d_output, learning_rate, mode=mode, momentum=momentum,
                                              decay_rate=decay_rate, l2_lambda=l2_lambda)

            train_loss_avg = epoch_loss / n_samples
            history['loss'].append(train_loss_avg)
            for idx, layer in enumerate(self.layers):
                history['weights'][idx].append(layer.weights.copy())

            self.pbar.update(1)

            if validation_data is not None:
                X_val, y_val = validation_data
                val_output = self.predict(X_val)

                if self.task == "classification":
                    val_loss = -np.sum(y_val * np.log(val_output + 1e-9)) / len(X_val)
                else:
                    val_loss = np.mean((y_val - val_output) ** 2)

                history['val_loss'].append(val_loss)
                self.pbar.set_postfix({'loss': f"{train_loss_avg:.4f}", 'val_loss': f"{val_loss:.4f}"})

                if patience is not None:
                    if val_loss < self._es_state['best_val_loss']:
                        self._es_state['best_val_loss'] = val_loss
                        self._es_state['patience_counter'] = 0
                        self._es_state['best_weights'] = [layer.weights.copy() for layer in self.layers]
                        self._es_state['best_biases'] = [layer.biases.copy() for layer in self.layers]
                    else:
                        self._es_state['patience_counter'] += 1
                        if self._es_state['patience_counter'] >= patience:
                            tqdm.write(
                                f"\n[Early Stopping] Aktywowano! Najlepszy błąd walidacji: {self._es_state['best_val_loss']:.6f}")
                            for idx, layer in enumerate(self.layers):
                                layer.weights = self._es_state['best_weights'][idx]
                                layer.biases = self._es_state['best_biases'][idx]

                            self.pbar.close()
                            self.pbar = None
                            self._es_state = None
                            break
            else:
                self.pbar.set_postfix({'loss': f"{train_loss_avg:.4f}"})

        if hasattr(self, 'pbar') and self.pbar is not None and self.pbar.n >= self.pbar.total:
            self.pbar.close()
            self.pbar = None
            self._es_state = None

        return history

    def fit_with_loss_history(self, X, y, epochs, learning_rate, batch_size=None, mode="simple", momentum=0.9,
                              decay_rate=0.9, l2_lambda=0.0, validation_data=None, patience=None, tqdm_desc="Trening",
                              total_epochs=None):
        n_samples = X.shape[0]
        if batch_size is None:
            batch_size = n_samples

        history = {'loss': [], 'val_loss': []}

        actual_total = total_epochs if total_epochs is not None else epochs

        if isinstance(patience, float):
            patience = max(1, int(patience * actual_total))

        if not hasattr(self, 'pbar') or self.pbar is None:
            self.pbar = tqdm(total=actual_total, desc=tqdm_desc, leave=True)
            self._es_state = {
                'best_val_loss': float('inf'),
                'patience_counter': 0,
                'best_weights': None,
                'best_biases': None
            }

        for epoch in range(epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0

            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                output = self.predict(X_batch)

                if self.task == "classification":
                    loss = -np.sum(y_batch * np.log(output + 1e-9)) / len(X_batch)
                    d_output = (output - y_batch) / len(X_batch)
                else:
                    loss = np.mean((y_batch - output) ** 2)
                    d_output = 2 * (output - y_batch) / len(X_batch)

                epoch_loss += loss * len(X_batch)

                for layer in reversed(self.layers):
                    d_output = layer.backward(d_output, learning_rate, mode=mode, momentum=momentum,
                                              decay_rate=decay_rate, l2_lambda=l2_lambda)

            train_loss_avg = epoch_loss / n_samples
            history['loss'].append(train_loss_avg)

            self.pbar.update(1)

            if validation_data is not None:
                X_val, y_val = validation_data
                val_output = self.predict(X_val)

                if self.task == "classification":
                    val_loss = -np.sum(y_val * np.log(val_output + 1e-9)) / len(X_val)
                else:
                    val_loss = np.mean((y_val - val_output) ** 2)

                history['val_loss'].append(val_loss)
                self.pbar.set_postfix({'loss': f"{train_loss_avg:.4f}", 'val_loss': f"{val_loss:.4f}"})

                if patience is not None:
                    if val_loss < self._es_state['best_val_loss']:
                        self._es_state['best_val_loss'] = val_loss
                        self._es_state['patience_counter'] = 0
                        self._es_state['best_weights'] = [layer.weights.copy() for layer in self.layers]
                        self._es_state['best_biases'] = [layer.biases.copy() for layer in self.layers]
                    else:
                        self._es_state['patience_counter'] += 1
                        if self._es_state['patience_counter'] >= patience:
                            tqdm.write(
                                f"\n[Early Stopping] Aktywowano! Najlepszy błąd walidacji: {self._es_state['best_val_loss']:.6f}")
                            for idx, layer in enumerate(self.layers):
                                layer.weights = self._es_state['best_weights'][idx]
                                layer.biases = self._es_state['best_biases'][idx]

                            self.pbar.close()
                            self.pbar = None
                            self._es_state = None
                            break
            else:
                self.pbar.set_postfix({'loss': f"{train_loss_avg:.4f}"})

        if hasattr(self, 'pbar') and self.pbar is not None and self.pbar.n >= self.pbar.total:
            self.pbar.close()
            self.pbar = None
            self._es_state = None

        return history