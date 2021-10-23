
enum Func{
    TAHN=1,
    RELU=2,
    SIGM=3
};

class Neuron{
public:
    Neuron(double value);
    Neuron(double value, Func funcActivated);

    void setValue(double value);
    void activate();
    void derive();

    double getValue(){ return this->value; }
    double getActivateValue(){ return this->activatedValue; }
    double getDerivedValue(){ return this->derivedValue; }

private:
    double value;
    double activatedValue;
    double derivedValue;

    Func funcType = Func::RELU;
};
