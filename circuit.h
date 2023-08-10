#ifndef __CIRCUIT_H__
#define __CIRCUIT_H__

#include <functional>
#include "field.h"

class Gate {
    public:
        enum Type {ADD, MUL, SUB, XOR, OR, NAAB,
            NOT, SUM, EXPSUM, GEXPSUM, IN, DUMMY, MIMC, LINEAR, SCALAR,
            AES,
            type_count = EXPSUM,
            AND = MUL, BITTEST = NAAB, RELAY = SUM, ONESUB = NOT};
        // NAAB: not A and B
        // BITTEST and RELAY gates:
        // call init(type, int) to force input[0] == input[1]
        // BITTEST(u) = NAAB(u, u) = u (1 - u)
        // RELAY(u) = SUM(u, u) = u
        Type type;
        int input[2];
        int util[3];

        Gate() : type(DUMMY) {
            input[0] = 0;
            input[1] = 0;
        }

        void init(Type, int = 0);
        void init(Type, int, int);
        void init(Type, int, int, int, int = 0, int = -1);
};

std::ostream& operator<<(std::ostream&, Gate&);

template<class F> class Layer {
    public:
        Gate *gates;
        F *values;
        int bit_len;
        int grp_len;
        int phase_num;
        int degree;
        int max_degree; // max degree of all the gate types of this layer

        Layer() : phase_num(2), degree(1), max_degree(1) {}
        ~Layer();

        void init(int, int = 0, int = 2, int = 1, int = 1);
};

template<class F> std::ostream& operator<<(std::ostream&, Layer<F>&);


template<class F> class Circuit {
    public:
        Layer<F> *layers;
        typedef std::function<F *(F *, F **, F **)> PredFunc;
        typedef std::function<F (F *, F **)> PredVFunc;
        PredFunc *predicates;
        PredVFunc *predVs;
        int depth;
        F *inputs;

        int secret_len;
        // F mimc_key;

        Circuit() {}

        Circuit(int);

        ~Circuit();

        void init(int);
        void read(const char*);

        F *predicate(int layer, F *scalar, F **r0, F **r);

        void eval(F *);

        void get_values(int, F *) const;

        // virtual void print() const;
};

// template<class F> std::ostream& operator<<(std::ostream&, Circuit<F>&);

#endif /* __CIRCUIT_H__ */
