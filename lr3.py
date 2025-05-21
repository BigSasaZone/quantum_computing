import numpy as np
from abc import ABC, abstractmethod

class Qubit:
    def __init__(self, alpha=1.0, beta=0.0):
        self.state = np.array([alpha, beta], dtype=complex)
    
    def apply_gate(self, gate):
        self.state = np.dot(gate, self.state)
    
    def measure(self):
        probabilities = np.abs(self.state)**2
        result = np.random.choice([0, 1], p=probabilities)
        self.state = np.zeros(2, dtype=complex)
        self.state[result] = 1.0
        return result, probabilities[result]
    
    def get_probabilities(self):
        return np.abs(self.state)**2
    
    def __str__(self):
        return f"{self.state[0]:.4f}|0⟩ + {self.state[1]:.4f}|1⟩"

class QuantumRegister:
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.state = np.zeros(2**num_qubits, dtype=complex)
        self.state[0] = 1.0
    
    def apply_single_gate(self, gate, target):
        operators = []
        for q in range(self.num_qubits):
            operators.append(gate if q == target else np.eye(2, dtype=complex))
        full_matrix = operators[0]
        for op in operators[1:]:
            full_matrix = np.kron(full_matrix, op)
        self.state = np.dot(full_matrix, self.state)
    
    def apply_controlled_gate(self, gate, control, target):
        dim = 2**self.num_qubits
        controlled_matrix = np.eye(dim, dtype=complex)
        for i in range(dim):
            bits = [(i >> j) & 1 for j in range(self.num_qubits)]
            if bits[control] == 1:
                new_bits = bits.copy()
                target_state = np.array([1-bits[target], bits[target]], dtype=complex)
                new_target_state = np.dot(gate, target_state)
                if new_target_state[0] != 0:
                    new_bits[target] = 0
                    j = sum([bit << k for k, bit in enumerate(reversed(new_bits))])
                    controlled_matrix[j, i] = new_target_state[0]
                if new_target_state[1] != 0:
                    new_bits = bits.copy()
                    new_bits[target] = 1
                    j = sum([bit << k for k, bit in enumerate(reversed(new_bits))])
                    controlled_matrix[j, i] = new_target_state[1]
                if new_target_state[0] != 0 or new_target_state[1] != 0:
                    controlled_matrix[i, i] = 0
        self.state = np.dot(controlled_matrix, self.state)
    
    def apply_x(self, target):
        x_gate = np.array([[0, 1], [1, 0]], dtype=complex)
        self.apply_single_gate(x_gate, target)
    
    def apply_y(self, target):
        y_gate = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self.apply_single_gate(y_gate, target)
    
    def apply_z(self, target):
        z_gate = np.array([[1, 0], [0, -1]], dtype=complex)
        self.apply_single_gate(z_gate, target)
    
    def apply_hadamard(self, target):
        h_gate = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        self.apply_single_gate(h_gate, target)
    
    def apply_cnot(self, control, target):
        x_gate = np.array([[0, 1], [1, 0]], dtype=complex)
        self.apply_controlled_gate(x_gate, control, target)
    
    def apply_cz(self, control, target):
        z_gate = np.array([[1, 0], [0, -1]], dtype=complex)
        self.apply_controlled_gate(z_gate, control, target)
    
    def apply_swap(self, qubit1, qubit2):
        self.apply_cnot(qubit1, qubit2)
        self.apply_cnot(qubit2, qubit1)
        self.apply_cnot(qubit1, qubit2)
    
    def measure(self, target=None):
        if target is not None:
            probabilities = np.zeros(2)
            for i in range(2**self.num_qubits):
                bit_value = (i >> target) & 1
                probabilities[bit_value] += abs(self.state[i])**2
            result = np.random.choice([0, 1], p=probabilities)
            new_state = np.zeros(2**self.num_qubits, dtype=complex)
            norm = 0
            for i in range(2**self.num_qubits):
                bit_value = (i >> target) & 1
                if bit_value == result:
                    new_state[i] = self.state[i]
                    norm += abs(self.state[i])**2
            if norm > 0:
                new_state /= np.sqrt(norm)
            self.state = new_state
            return result
        else:
            probabilities = self.get_probabilities()
            result = np.random.choice(2**self.num_qubits, p=probabilities)
            self.state = np.zeros(2**self.num_qubits, dtype=complex)
            self.state[result] = 1.0
            return result
    
    def get_probabilities(self):
        return np.abs(self.state)**2
    
    def __str__(self):
        result = []
        for i in range(2**self.num_qubits):
            if abs(self.state[i]) > 1e-10:
                bits = format(i, f'0{self.num_qubits}b')
                result.append(f"{self.state[i]:.4f}|{bits}⟩")
        return " + ".join(result) if result else "0"

class Oracle(ABC):
    @abstractmethod
    def apply(self, qreg): pass
    
    @abstractmethod
    def description(self): pass

class ConstantOracle(Oracle):
    def __init__(self, value=1):
        self.value = value
    
    def apply(self, qreg):
        if self.value == 1:
            qreg.apply_x(qreg.num_qubits - 1)
    
    def description(self):
        return f"f(x) = {self.value} (постоянная функция)"

class BalancedOracle(Oracle):
    def __init__(self, input_qubits):
        self.input_qubits = input_qubits
    
    def apply(self, qreg):
        for qubit in self.input_qubits:
            qreg.apply_cnot(qubit, qreg.num_qubits - 1)
    
    def description(self):
        qubits_str = " ⊕ ".join([f"x{i}" for i in self.input_qubits])
        return f"f(x) = {qubits_str} (сбалансированная функция)"

class GroverOracle(Oracle):
    def __init__(self, target_state):
        if isinstance(target_state, str):
            self.target = int(target_state, 2)
            self.num_qubits = len(target_state)
        else:
            self.target = target_state
            self.num_qubits = max(1, (self.target.bit_length()))
    
    def apply(self, qreg):
        oracle_matrix = np.eye(2**qreg.num_qubits, dtype=complex)
        oracle_matrix[self.target, self.target] = -1
        qreg.state = np.dot(oracle_matrix, qreg.state)
    
    def description(self):
        bits = format(self.target, f'0{self.num_qubits}b')
        return f"Оракул Гровера для состояния |{bits}⟩"

class AndOracle(GroverOracle):
    def __init__(self, qubits):
        self.qubits = qubits
        target = 0
        for q in qubits:
            target |= (1 << q)
        super().__init__(target)
    
    def description(self):
        qubits_str = " ∧ ".join([f"x{i}" for i in self.qubits])
        return f"f(x) = {qubits_str} = 1"

class GroverAlgorithm:
    def __init__(self, num_qubits, oracle):
        self.num_qubits = num_qubits
        self.oracle = oracle
        self.qreg = QuantumRegister(num_qubits)
        for q in range(num_qubits):
            self.qreg.apply_hadamard(q)
    
    def apply_diffusion(self):
        for q in range(self.num_qubits):
            self.qreg.apply_hadamard(q)
        zero_reflection = np.eye(2**self.num_qubits, dtype=complex)
        zero_reflection[0, 0] = 1
        for i in range(1, 2**self.num_qubits):
            zero_reflection[i, i] = -1
        self.qreg.state = np.dot(zero_reflection, self.qreg.state)
        for q in range(self.num_qubits):
            self.qreg.apply_hadamard(q)
    
    def run(self, iterations=None):
        if iterations is None:
            N = 2**self.num_qubits
            iterations = int(np.round(np.pi/4 * np.sqrt(N)))
        for _ in range(iterations):
            self.oracle.apply(self.qreg)
            self.apply_diffusion()
        probabilities = self.qreg.get_probabilities()
        max_prob_state = np.argmax(probabilities)
        max_prob = probabilities[max_prob_state]
        return max_prob_state, max_prob
    
    def visualize_probabilities(self):
        probabilities = self.qreg.get_probabilities()
        result = []
        for i in range(2**self.num_qubits):
            if probabilities[i] > 1e-10:
                bits = format(i, f'0{self.num_qubits}b')
                result.append(f"|{bits}⟩: {probabilities[i]:.4f}")
        return "\n".join(result)

def test_quantum_gates():
    print("=== Тест квантовых гейтов ===")
    print("\nТест однокубитных гейтов:")
    qubit = Qubit()
    print(f"Начальное состояние: {qubit}")
    x_gate = np.array([[0, 1], [1, 0]], dtype=complex)
    qubit.apply_gate(x_gate)
    print(f"После X-гейта: {qubit}")
    h_gate = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    qubit = Qubit()
    qubit.apply_gate(h_gate)
    print(f"После H-гейта: {qubit}")
    print("\nТест CNOT-гейта:")
    qreg = QuantumRegister(2)
    print(f"Начальное состояние: {qreg}")
    qreg.apply_hadamard(0)
    qreg.apply_cnot(0, 1)
    print(f"Состояние Белла: {qreg}")
    print(f"Вероятности: {qreg.get_probabilities()}")

def test_oracles():
    print("\n=== Тест оракулов ===")
    print("\nТест постоянного оракула (f(x) = 1):")
    qreg = QuantumRegister(3)
    qreg.apply_hadamard(0)
    qreg.apply_hadamard(1)
    qreg.apply_x(2)
    qreg.apply_hadamard(2)
    print(f"Состояние до оракула: {qreg}")
    oracle = ConstantOracle(1)
    oracle.apply(qreg)
    print(f"Состояние после оракула: {qreg}")
    print(f"Описание оракула: {oracle.description()}")
    print("\nТест сбалансированного оракула (f(x) = x0 ⊕ x1):")
    qreg = QuantumRegister(3)
    qreg.apply_hadamard(0)
    qreg.apply_hadamard(1)
    qreg.apply_x(2)
    qreg.apply_hadamard(2)
    print(f"Состояние до оракула: {qreg}")
    oracle = BalancedOracle([0, 1])
    oracle.apply(qreg)
    print(f"Состояние после оракула: {qreg}")
    print(f"Описание оракула: {oracle.description()}")

def test_grover():
    print("\n=== Тест алгоритма Гровера ===")
    oracle = AndOracle([0, 1])
    grover = GroverAlgorithm(2, oracle)
    print(f"Описание оракула: {oracle.description()}")
    print("\nНачальные вероятности:")
    print(grover.visualize_probabilities())
    result, probability = grover.run(iterations=1)
    print("\nПосле 1 итерации:")
    print(grover.visualize_probabilities())
    bits = format(result, f'0{grover.num_qubits}b')
    print(f"\nНайденное состояние: |{bits}⟩ с вероятностью {probability:.4f}")

if __name__ == "__main__":
    test_quantum_gates()
    test_oracles()
    test_grover()
