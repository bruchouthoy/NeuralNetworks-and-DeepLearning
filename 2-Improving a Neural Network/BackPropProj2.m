%Brunda Chouthoy
%CSC578 - Project 2
%Improving a Neural Network
%Oct 22, 2017

function [weight, bias, acc, cost] = BackPropProj2(inputs, targets, nodeLayers, numEpochs, batchSize, eta, split, momentum, lambda, transFunction, costFun)
  %Partition dataset into train, test and validation sets using split attribute
  if (isempty(split))
      trainInputs = inputs;
      testInputs = [];
      valInputs = [];
      trainTargets = targets;
      testTargets = [];
      valTargets = [];
  else
      % using the dividerand function to split - [trainInd,valInd,testInd] = dividerand(Q,trainRatio,valRatio,testRatio)
      [trainInd, valInd, testInd] = dividerand((size(inputs,2)), split(1)/100, split(2)/100, split(3)/100);
      trainInputs = inputs(:, trainInd);
      valInputs = inputs(:, valInd);
      testInputs = inputs(:, testInd);
      trainTargets = targets(:, trainInd);
      valTargets = targets(:, valInd);
      testTargets = targets(:, testInd);
  end

  %Initializations
  L = size(nodeLayers,2); %total number of node layers
  weight = cell(1,L); %Initialize a cell to hold the weight
  bias = cell(1,L); %Initialize a cell to hold the biases
  bigDeltaWeight = {}; %Initialize a cell to hold deltas for the weights
  bigDeltaBias = {}; %Initialize a cell to hold deltas for the bias
  trainInputSize = size(trainInputs,2);%Hold size of the train inputs
  valInputSize = size(valInputs,2);%Hold size of the validation inputs
  testInputSize = size(testInputs,2);%Hold size of the test inputs
  batchValues = {}; %Initialize a cell to hold batch values
  targetValues = {}; %Initialize a cell to hold target values
  batchIndex = 1; %Initialize a counter variable
  %To store costs for training,testing and validation sets
  trainCost = zeros(1,numEpochs);
  testCost = zeros(1,numEpochs);
  valCost = zeros(1,numEpochs);
  %Initialize cells for accuracy and cost to hold the final train,validation and test values 
  acc = {};
  cost = {};

  for layer = 2 : L
    weight{layer} = randn(nodeLayers(layer), nodeLayers(layer-1))/sqrt(nodeLayers(layer-1));
    bias{layer} = randn(nodeLayers(layer), 1);
  end

  %Dividing the input matrix into mini batches
  %Increment by the value batchSize(step) for each iteration
  for initPos = 1:batchSize:trainInputSize
    if trainInputSize - initPos >= batchSize
        miniBatch = trainInputs(:, initPos:initPos+batchSize-1);
        batchValues{batchIndex} = miniBatch;
        target = trainTargets(:, initPos:initPos+batchSize-1);
        targetValues{batchIndex} = target;
        batchIndex = batchIndex + 1;
    else
        miniBatch = trainInputs(:, initPos:end);
        batchValues{batchIndex} = miniBatch;
        target = trainTargets(:, initPos:end);
        targetValues{batchIndex} = target;
    end 
  end

  %To diaplay output headers
  fprintf('|        TRAIN           ||      VALIDATION          ||          TEST\n');
  fprintf('------------------------------------------------------------------------------------------\n');
  fprintf('Epoch  | Cost | Corr  |  Acc  ||  Cost |  Corr |  Acc ||  Cost | Corr |  Acc \n');
  fprintf('------------------------------------------------------------------------------------------\n');

  %Displaying user input for 'softmax' and 'relu' transfer functions
  if (strcmp(transFunction, 'softmax'))
      userInSoftMax = input('User Input required: Softmax function can only be used in the last layer \n');
  elseif (strcmp(transFunction,'relu'))
      userInReLu = input('User Input required: Relu can only be used in the hidden layers \n');
  end
  
  function a = transFunApply(input)
    if (strcmp(transFunction, 'sigmoid'))
        a = transfer(input, transFunction);
    elseif (strcmp(transFunction, 'tanh'))
        if layer == L
            a = transfer(input, transFunction); 
        else
            a = tanh(input);
        end
    elseif (strcmp(transFunction, 'relu'))
        if layer == L
            a = transfer(input, userInReLu);%ReLu cannot be used in the last layer.
        else
            a = max(0, input);
        end
    elseif (strcmp(transFunction, 'softmax'))
        if layer == L
            a = transfer(input, transFunction); 
        else
            a = transfer(input, userInSoftMax);%Cannot use softmax if it's the last layer 
        end
    else
        error('Not a valid Transfer function. Input sigmoid,tanh,relu or softmax')
    end
  end
  
  %Loop through each epoch and batch
  for epoch = 1:numEpochs
      %Using the randperm function for minibatch shuffling
      random = randperm(size(batchValues,2));
      batchCounter = 1;
      for batch = 1:size(batchValues,2)
          z = {}; %Initialize a cell to hold values for the intermediate nodes
          a = {}; %Initialize a cell to hold the activation function
          a{1} = batchValues{random(batchCounter)}; %Input value of the batch is assigned to the first element of the activation cell

          %Feedforward the network - For each layer calculate z{layer} and a{layer}
          for layer = 2 : L
             z{layer} = weight{layer} * a{layer - 1} + bias{layer};
             a{layer} = transFunApply(z{layer});
          end

          delta = {}; %Initialize a cell to hold the error values
	      error = (a{L} - targetValues{random(batch)});

          %Calculating the output layer Error for each activation function
          if (strcmp(transFunction, 'tanh'))
            delta{L} = error .* (1-tanh(z{L}).^2);
          elseif (strcmp(transFunction, 'sigmoid'))
            delta{L} = error .* SigmoidPrime(z{L});
          elseif (strcmp(transFunction, 'softmax'))
            %If softmax is the trans function for the last output layer
            delta{L} = error .* ones(size(z{L}));
          elseif (strcmp(transFunction, 'relu')) 
            if (strcmp(userInReLu, 'softmax'))  
                delta{L} = error .* ones(size(z{L}));
            else 
                delta{L} = error .* (dTransferPrime(z{L}, userInReLu)); 
            end
          end
          
          % Back propagate error through the network from L to 2nd layer
          % Calculating Hidden layer errors 
          for layer = (L - 1) : -1 : 2
              if (strcmp(transFunction, 'softmax'))%If softmax is the transfer function - it cannot be used in hidden layers
                delta{layer} = (weight{layer + 1}.' * delta{layer + 1}) .* dTransferPrime(z{layer}, userInSoftMax);
              else
                delta{layer} = (weight{layer + 1}.' * delta{layer + 1}) .* dTransferPrime(z{layer}, transFunction);
              end
          end
            
         %Gradient Descent and finding the minimum
         %For each layer from L to 2nd layer
          for layer = L : -1 : 2
              if epoch == 1 && batch == 1 
                  weight{layer} = weight{layer} - eta/length(batchValues{random(batch)}) * delta{layer} * a{layer - 1}.'; 
                  bias{layer} = bias{layer} - eta/length(batchValues{random(batch)}) * sum(delta{layer}, 2);
                  bigDeltaWeight{layer} = eta/length(batchValues{random(batch)}) * delta{layer} * a{layer - 1}.';
                  bigDeltaBias{layer} = eta/length(batchValues{random(batch)}) * sum(delta{layer}, 2);
              else %Using the momentum parameter to find the minimum coefficients
                  weight{layer} = weight{layer} + bigDeltaWeight{layer};
                  bias{layer} = bias{layer} + bigDeltaBias{layer};
                  bigDeltaWeight{layer} = momentum .* bigDeltaWeight{layer} - eta/length(batchValues{random(batch)}) * delta{layer} * a{layer - 1}.';
                  bigDeltaBias{layer} = momentum .* bigDeltaBias{layer} - eta/length(batchValues{random(batch)}) * sum(delta{layer}, 2);
              end
          end
          batchCounter = batchCounter + 1;
      end 

      %Compute final output values after updating weights
      trainOut = {}; %Initialize a cell to hold the final training output values 
      trainOut{1} = trainInputs; %Assign inputs to first element of the output cell
      valOut = {}; %Initialize a cell to hold the final validation output values 
      valOut{1} = valInputs;
      testOut = {}; %Initialize a cell to hold the final testing output values 
      testOut{1} = testInputs;
      
      ztrain = cell(1,L);
      zval = cell(1,L);
      ztest = cell(1,L);
      weightSum = 0;
      %For each layer from the 2nd layer
      for layer = 2 : L
          ztrain{layer} = (weight{layer} * trainOut{layer-1}) + (bias{layer});
          zval{layer} = (weight{layer} * valOut{layer-1}) + (bias{layer});
          ztest{layer} = (weight{layer} * testOut{layer-1}) + (bias{layer});
          
          trainOut{layer} = transFunApply(ztrain{layer});
          valOut{layer} = transFunApply(zval{layer});
          testOut{layer} = transFunApply(ztest{layer});
          
          weightSum = weightSum + sum(sum(weight{layer}.^2));
      end
       
      L2Train = lambda/(2*trainInputSize) * weightSum;
      L2Val = lambda/(2*valInputSize) * weightSum;
      L2Test = lambda/(2*testInputSize) * weightSum;

      %Compute the number of correct cases
      %correct = correct + sum(all(targets==round(output{L}),1),2);
      trainCorrect = sum(all(trainTargets == round(trainOut{L}),1),2);
      valCorrect = sum(all(valTargets == round(valOut{L}),1),2);
      testCorrect = sum(all(testTargets == round(testOut{L}),1),2);
   
      %Computing train, validation and test set accuracies
      trainAccuracy = trainCorrect/trainInputSize; 
      valAccuracy = valCorrect/valInputSize;
      testAccuracy = testCorrect/testInputSize;

      % Computing cost 
      if (strcmp(costFun,'quad')) %Quadratic
          trainCost = 1/(2*trainInputSize) * sum(sum((0.5*(trainTargets - trainOut{L}).^2)))+L2Train;
          valCost = 1/(2*valInputSize) * sum(sum((0.5*(valTargets - valOut{L}).^2)))+L2Val;
          testCost = 1/(2*testInputSize) * sum(sum((0.5*(testTargets - testOut{L}).^2)))+L2Test;
      elseif (strcmp(costFun,'cross')) %Cross-Entropy
          trainCost = -1/(trainInputSize) .* sum(sum(trainTargets .* log(trainOut{L}+eps) + (1-trainTargets) .* log(1-trainOut{L}))+eps)+L2Train;
          valCost = -1/(valInputSize) .* sum(sum(valTargets .* log(valOut{L}+eps) + (1-valTargets) .* log(1-valOut{L}))+eps)+L2Val;
          testCost = -1/(testInputSize) .* sum(sum(testTargets .* log(testOut{L}+eps) + (1-testTargets) .* log(1-testOut{L}))+eps)+L2Test;
      elseif (strcmp(costFun,'log')) %log-likelihood
          trainCost = sum(-log(max(trainOut{L})+eps)/trainInputSize)+L2Train;
          valCost = sum(-log(max(valOut{L})+eps)/valInputSize)+L2Val;
          testCost = sum(-log(max(testOut{L})+eps)/testInputSize)+L2Test;
      end    
      
      % store costs for early stopping 
      trainCost(epoch) = trainCost;
      valCost(epoch) = valCost;
      testCost(epoch) = testCost;
      
      fprintf('%d\t| %.5f | %d/%d | %.5f || %.5f | %d/%d | %.5f || %.5f | %d/%d | %.5f\n', ...
      epoch,trainCost(epoch),trainCorrect,trainInputSize,trainAccuracy,...
      valCost(epoch),valCorrect,valInputSize,valAccuracy,...
      testCost(epoch),testCorrect,testInputSize,testAccuracy);

      %Early stopping conditions and plots  
      %Storing accuracy and costs for each epoch
      acc{1}(epoch) = trainAccuracy;
      acc{2}(epoch) = valAccuracy;
      acc{3}(epoch) = testAccuracy; 
      cost{1}(epoch) = trainCost(epoch);
      cost{2}(epoch) = valCost(epoch);
      cost{3}(epoch) = testCost(epoch);

      %Early Stopping based on Accuracy -- If all the input cases are correct - accuracy=1
      if trainCorrect == trainInputSize && valCorrect == valInputSize && testCorrect == testInputSize
          fprintf('Accuracy is 1 and all the cases are identified correct - Early stopping \n');
          subplot(2,1,1)
          plot(cost{1}); hold on; plot(cost{2});plot(cost{3}); 
          title('Cost plot'); xlabel('Number of epochs'); ylabel('cost'); 
          legend('Training cost','Validation cost', 'Testing cost');hold off;
          
          subplot(2,1,2);
          plot(acc{1});hold on;plot(acc{2});plot(acc{3}); 
          title('Accuracy plot'); xlabel('Number of epochs'); ylabel('Accuracy'); 
          legend('Training acc', 'Validation acc', 'Testing acc');hold off;
          break
      %Early stopping strategy - check if  Validation cost increases when the number of epochs is greater than 65% of the epochs 
      elseif epoch > round(numEpochs*0.65)
          if valCost(epoch) > valCost(epoch-1)
              subplot(2,1,1);
              fprintf('Validation cost increased after 65 percent of epochs were executed -- Early stopping criteria \n');
              plot(cost{1}); hold on;plot(cost{2}); plot(cost{3}); 
              title('Cost plot'); xlabel('Number of epochs'); ylabel('cost'); 
              legend('Training cost','Validation cost', 'Testing cost');hold off;
                
              subplot(2,1,2);
              plot(acc{1}); hold on;plot(acc{2}); plot(acc{3}); 
              title('Accuracy plot'); xlabel('Number of epochs'); ylabel('Accuracy'); 
              legend('Training acc', 'Validation acc', 'Testing acc');hold off;
          break
          end
      else
          subplot(2,1,1);
          plot(cost{1});hold on;plot(cost{2}); plot(cost{3});
          title('Cost plot'); xlabel('Number of epochs'); ylabel('cost'); 
          legend('Training cost','Validation cost', 'Testing cost');hold off;

          subplot(2,1,2);
          plot(acc{1});hold on;plot(acc{2}); plot(acc{3}); 
          title('Accuracy plot'); xlabel('Number of epochs'); ylabel('Accuracy'); 
          legend('Training acc', 'Validation acc', 'Testing acc');hold off;
      end
  end
end






