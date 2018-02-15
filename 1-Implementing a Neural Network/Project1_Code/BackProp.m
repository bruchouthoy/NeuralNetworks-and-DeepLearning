%Brunda Chouthoy
%CSC578 - Project 1
%Implementing a Neural Network - Backpropogation algorithm

function [ weight, bias ] = BackProp(inputs, targets, nodeLayers, numEpochs, batchSize, eta) 
  L = size(nodeLayers,2); %total number of node layers
  weight = cell(1,L); %Initialize a cell to hold the weight
  bias = cell(1,L); %Initialize a cell to hold the biases
  batchValues = {}; %Initialize a cell to hold batch values
  targetValues = cell(1,L); %Initialize a cell to hold target values
  batchIndex = 1; %Initialize a counter variable
  inputSize = size(inputs,2); %total number of inputs

  %For each layer from the 2nd layer
  for layer = 2:L
      %Initialize weight and biases with randomized normal distribution values i.e mean=0 and SD=1
      weight {layer} = randn(nodeLayers(layer), nodeLayers(layer-1));
      bias{layer} = randn(nodeLayers(layer), 1);
  end
  
  %Dividing the input matrix into mini batches
  %Increment by the value batchSize(step) for each iteration
  for initPos = 1:batchSize:inputSize
    if inputSize - initPos >= batchSize
        miniBatch = inputs(:, initPos: initPos + batchSize - 1);
        batchValues{batchIndex} = miniBatch;
        target = targets(:, initPos: initPos + batchSize - 1);
        targetValues{batchIndex} = target;
        batchIndex = batchIndex + 1;
    else
        miniBatch = inputs(:, initPos:end);
        batchValues{batchIndex} = miniBatch;
        target = targets(:, initPos:end);
        targetValues{batchIndex} = target;
    end 
  end 

  %Loop through each epoch and batch
  for epoch = 1:numEpochs
      for batch = 1:size(batchValues,2)
          z = cell(1,L); %Initialize a cell to hold values for the intermediate nodes
          a = cell(1,L); %Initialize a cell to hold the activation function
          a{1} = batchValues{batch}; %Input value of the batch is assigned to the first element of the activation cell
          correct = 0;
     
          %Feedforward the network - For each layer calculate z{layer} and a{layer}
          for layer = 2:L
              z{layer} = weight{layer} * a{layer - 1} + bias{layer};
              a{layer} = Sigmoid(z{layer});
          end
          
          % Calculate the output layer error i.e. delta
          delta = cell(1,L); %Initialize a cell to hold the error values
          % Invoke the SigmoidPrime function
	      error = (a{L} - targetValues{batch});
          cost = error .* SigmoidPrime(z{L});
          delta{L} = cost;
           
          % Back propagate error through the network from L to 2nd layer
          % Step size is -1 
          for layer = (L - 1) : -1 : 2
              delta{layer} = (weight{layer + 1}.' * delta{layer + 1}) .* SigmoidPrime(z{layer});
          end
            
         %Gradient Descent and finding the minimum
         %For each layer from L to 2nd layer
          for layer = L : -1 : 2
              weight{layer} = weight{layer} - (eta/length(batchValues{batch})) * (delta{layer} * a{layer - 1}.');
              bias{layer} = bias{layer} - (eta/length(batchValues{batch})) * sum(delta{layer}, 2);
          end
      end 
      
      % Compute final output values after updating weights
      output = {}; %Initialize a cell to hold the final output values 
      output{1} = inputs; %Assign inputs to first element of the output cell
      %For each layer from the 2nd layer
      for layer = 2 : L
          z1 = (weight{layer}*output{layer-1})+(bias{layer});
          output{layer} = logsig(z1); %hold the activation for next layer
      end
      error = output{L}-targets;
      %Computing the value of MSE .. Diving by 2n
      MSE = sqrt(sum(sum(error.^2)))/(2*inputSize);
      
      %Compute the number of correct cases
      correct = correct + sum(all(targets==round(output{L}),1),2);
      
      %Computing accuracy
      accuracy = correct/inputSize;
      
     %Reporting based on Accuracy
     %If all the input cases are correct - accuracy=1
      if correct == inputSize 
      	fprintf('Epoch %d, MSE: %f, Correct: %d/%d, Acc: %f \n', epoch, MSE, correct, inputSize, accuracy);
        break
      end

      % Reporting for each iteration if the number of epochs are less than 100
      if numEpochs <= 100
      	fprintf('Epoch	%d, MSE: %f, Correct: %d / %d,  Acc: %f \n', epoch, MSE, correct, inputSize, accuracy);
      elseif mod(epoch, 100) == 0 && numEpochs > 100
      % if running a large number of epochs, reporting only the 100th epoch
      	fprintf('Epoch	%d, MSE: %f, Correct: %d / %d,  Acc: %f \n', epoch, MSE, correct, inputSize, accuracy);
      end
  end
  
  if mod(numEpochs, 100) ~= 0 && numEpochs > 100
  % Reporting for the last iteration
     fprintf('Epoch %d, MSE: %f, Correct: %d / %d,  Acc: %f \n', epoch, MSE, correct, inputSize, accuracy);
  end
end
