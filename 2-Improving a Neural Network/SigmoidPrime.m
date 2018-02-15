%Brunda Chouthoy
%CSC578 - Project 2
%Improving a Neural Network
%Oct 22, 2017

function dsigmoid = SigmoidPrime(x)
  %Derivative of the sigmoid function
  dsigmoid = Sigmoid(x).*(1 - Sigmoid(x));
end