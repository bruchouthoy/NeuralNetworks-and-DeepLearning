%Brunda Chouthoy
%CSC578 - Project 2
%Improving a Neural Network
%Oct 22, 2017

%dTransfer function finds derivative of the activation function
function df = dTransferPrime(z, fun) 
    if (strcmp(fun, 'sigmoid'))
        df=transfer(z,fun).*(1-transfer(z,fun)); 
    elseif (strcmp(fun, 'tanh'))
        df=1-transfer(z,fun).^2;
    elseif (strcmp(fun, 'relu'))
        df=double(z>0);
    elseif (strcmp(fun, 'softmax'))
        df=transfer(z,fun).*(1-transfer(z,fun)); 
    end
end