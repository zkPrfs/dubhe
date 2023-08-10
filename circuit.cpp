#include <iostream>
#include "circuit.h"

void Gate::init(Type t, int u) {
    type = t;
    input[0] = u;
    input[1] = u;
}

void Gate::init(Type t, int u, int v) {
    type = t;
    input[0] = u;
    input[1] = v;
}

void Gate::init(Type t, int u, int v, int x, int y, int z) {
    type = t;
    input[0] = u;
    input[1] = v;
    util[0] = x;
    util[1] = y;
    util[2] = z;
}

std::ostream& operator<<(std::ostream& os, Gate& g) {
    switch(g.type) {
    case Gate::ADD:
        os << "add(" << g.input[0] << "," << g.input[1] << ")";
        break;
    case Gate::MUL:
        os << "mul(" << g.input[0] << "," << g.input[1] << ")";
        break;
    case Gate::SUB:
        os << "sub(" << g.input[0] << "," << g.input[1] << ")";
        break;
    case Gate::XOR:
        os << "xor(" << g.input[0] << "," << g.input[1] << ")";
        break;
    case Gate::OR:
        os << "or(" << g.input[0] << "," << g.input[1] << ")";
        break;
    case Gate::NOT:
        os << "not(" << g.input[0] << ")";
        break;
    case Gate::NAAB:
        os << "naab(" << g.input[0] << "," << g.input[1] << ")";
        break;
    case Gate::SUM:
        os << "sum(" << g.input[0] << "," << g.input[1] << ")";
        break;
    case Gate::EXPSUM:
        os << "expsum(" << g.input[0] << "," << g.input[1] << ")";
        break;
    case Gate::GEXPSUM:
        os << "gexpsum(" << g.input[0] << "," << g.input[1] << ")";
        break;
    case Gate::IN:
        os << "in(" << g.input[0] << ")";
        break;
    case Gate::DUMMY:
        os << "dummy()";
        break;
    case Gate::MIMC:
        os << "mimc()";
        break;
    case Gate::LINEAR:
        os << "linear(" << g.input[0] << "," << g.input[1] << ")";
        break;
    case Gate::SCALAR:
        os << "scalar(" << g.input[0] << "," << g.input[1] << ")";
        break;
    default:
        break;
    }
    return os;
}

template<class F> void Layer<F>::init(int bit_len, int grp_len, int phase_num, int degree, int max_degree) {
    this->bit_len = bit_len;
    this->grp_len = grp_len;
    this->phase_num = phase_num;
    this->degree = degree;
    this->max_degree = max_degree;
    gates = new Gate[1 << bit_len];
    values = new F[1 << bit_len];
}

template<class F> Layer<F>::~Layer() {
    delete[] gates;
    delete[] values;
}

template<class F> std::ostream& operator<<(std::ostream &os, Layer<F> &layer) {
    for (int i = 0; i < 1 << layer.bit_len; i++)
        os << layer.gates[i] << "\t";
    return os;
}

template<class F> Circuit<F>::Circuit(int d) {
    depth = d;
    secret_len = 0;
    // mimc_key = F(0);
    layers = new Layer<F>[d+1];
    predicates = new PredFunc[d]();
    predVs = new PredVFunc[d]();
}

template<class F> void Circuit<F>::init(int d) {
    depth = d;
    secret_len = 0;
    // mimc_key = F(0);
    layers = new Layer<F>[d+1];
    predicates = new PredFunc[d]();
    predVs = new PredVFunc[d]();
}

template<class F> Circuit<F>::~Circuit() {
    delete[] layers;
    delete[] predicates;
    delete[] predVs;
}

// LONGTERM TODO implement predicate here, move R, S, R0, S0 into attrs of Circuit
template<class F> F *default_predicate(Layer<F> &layer, F *scalar, F **r0, F **r) {
    static F res[4];
    return res;
}

// LONGTERM TODO implement predicate here, move R, S, R0, S0 into attrs of Circuit
template<class F> F* Circuit<F>::predicate(int layer, F *scalar, F **r0, F **r) {
    if (predicates[layer - 1] != nullptr) {
        return ((predicates[layer - 1]))(scalar, r0, r);
    }
    else {
        return default_predicate<F>(layers[layer], scalar, r0, r);
    }
}


template<class F> void Circuit<F>::read(const char*) {
}

template<class F> void Circuit<F>::eval(F *inputs) {
    this->inputs = inputs;
    for (int j = 0; j <= depth; j++) {
        int bit_len = layers[j].bit_len;
        for (int i = 0; i < (1 << bit_len); i++) {
            int left = layers[j].gates[i].input[0];
            int right = layers[j].gates[i].input[1];
            switch (layers[j].gates[i].type) {
            case Gate::ADD:
                layers[j].values[i] =
                    layers[j-1].values[left] + layers[j-1].values[right];
                break;
            case Gate::MUL:
                layers[j].values[i] =
                    layers[j-1].values[left] * layers[j-1].values[right];
                break;
            case Gate::SUB:
                layers[j].values[i] =
                    layers[j-1].values[left] - layers[j-1].values[right];
                break;
            case Gate::XOR:
                layers[j].values[i] =
                    layers[j-1].values[left] + layers[j-1].values[right] -
                    layers[j-1].values[left] * layers[j-1].values[right] -
                    layers[j-1].values[left] * layers[j-1].values[right];
                break;
            case Gate::OR:
                layers[j].values[i] =
                    layers[j-1].values[left] + layers[j-1].values[right] -
                    layers[j-1].values[left] * layers[j-1].values[right];
                break;
            case Gate::NOT:
                layers[j].values[i] =
                    F(1) - layers[j-1].values[left];
                break;
            case Gate::NAAB:
                layers[j].values[i] = layers[j-1].values[right] *
                    (F(1) - layers[j-1].values[left]);
                break;
            case Gate::SUM:
                for (int u = left; u <= right; u++)
                    layers[j].values[i] += layers[j-1].values[u];
                break;
            case Gate::EXPSUM:
                for (int u = right; u >= left; u--)
                    layers[j].values[i] += layers[j].values[i] + layers[j-1].values[u];
                //  {
                //     layers[j].values[i] *= F(2);
                //     layers[j].values[i] += layers[j - 1].values[u];
                //  }
                break;
            case Gate::GEXPSUM:
                for (int u = right; u >= left; u--) {
                   layers[j].values[i] *= inputs[layers[j].gates[i].util[0]];
                   layers[j].values[i] += layers[j - 1].values[u];
                }
                break;
            case Gate::IN:
                layers[j].values[i] = inputs[left];
                break;
            case Gate::DUMMY:
                break;
            // case Gate::MIMC:
            //     layers[j].values[i] = (layers[j-1].values[i] + mimc_key + F::constant[j-1]).cube();
            //     if (j == F::round)
            //         layers[j].values[i] += mimc_key;
            //     break;
            case Gate::LINEAR:
                if (layers[j].gates[i].util[2] == -1)
                    layers[j].values[i] =
                        inputs[layers[j].gates[i].util[0]] *
                        layers[j-1].values[left] +
                        inputs[layers[j].gates[i].util[1]] *
                        layers[j-1].values[right];
                else
                    layers[j].values[i] =
                        inputs[layers[j].gates[i].util[0]] *
                        layers[j-1].values[left] +
                        inputs[layers[j].gates[i].util[1]] *
                        layers[j-1].values[right] +
                        inputs[layers[j].gates[i].util[2]];
                break;
            case Gate::SCALAR:
                layers[j].values[i] = layers[j-1].values[left] * inputs[right];
                break;
            case Gate::AES:
                layers[j].values[i] = layers[j-1].values[left] -
                    layers[j-1].values[left] * layers[j-1].values[left] * layers[j-1].values[right];
                break;
            default:
                std::cout << "WTF" << std::endl;
                break;
            }
        }
    }
}

template<class F> void Circuit<F>::get_values(int layer_id, F *res) const {
    int size = 1 << layers[layer_id].bit_len;
    for (int i = 0; i < size; i++)
        res[i] = layers[layer_id].values[i];
}

// template<class F> void Circuit<F>::print() const {
//     for (int d = 0; d <= depth; d++) {
//         for (int i = 0; i < (1 << layers[d].bit_len); i++)
//             std::cout << layers[d].values[i] << " ";
//         std::cout << std::endl;
//     }
// }

// template<class F> std::ostream& operator<<(std::ostream &os, Circuit<F> &C) {
//     for (int j = 0; j <= C.depth; j++) {
//         os << "layer " << j << ": ";
//         for (int i = 0; i < (1 << C.layers[j].bit_len); i++)
//             os << C.layers[j].gates[i] << "\t";
//         os << "\n";
//     }
//     return os;
// }

template class Circuit<field::GF2E>;
template class Layer<field::GF2E>;
// template std::ostream& operator<<(std::ostream &os, Layer<field::GF2E> &layer);
// template std::ostream& operator<<(std::ostream &os, Circuit<field::GF2E> &layer);
