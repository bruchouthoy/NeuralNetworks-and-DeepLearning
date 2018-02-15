%Brunda Chouthoy
%CSC578 - Project 2
%Improving a Neural Network
%Oct 22, 2017

function sig = Sigmoid(x)
    sig = 1./(1+exp(-x));
end