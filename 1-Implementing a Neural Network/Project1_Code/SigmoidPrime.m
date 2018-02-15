function dsigmoid = SigmoidPrime(x)
  %Derivative of the sigmoid function
  dsigmoid = Sigmoid(x).*(1 - Sigmoid(x));
end