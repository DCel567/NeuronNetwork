
enum Func{
    TAHN=1,
    RELU=2,
    SIGM=3
};

class Neuron{
public:
    Neuron(double value);
    Neuron(double value, Func funcActivated);

    void set_value(double value);
    void activate();
    void derive();

    double get_value(){}
    double get_activate_value(){ return this->output; }
    double get_derived_value(){ return this->derivedValue; }

private:
    double value;
    double activatedValue;
    double derivedValue;

    Func funcType = Func::RELU;
};
