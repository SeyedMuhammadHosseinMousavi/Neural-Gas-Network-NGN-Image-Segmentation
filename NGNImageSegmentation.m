% Image Segmentation and Quantization by Neural Gas Network (NGN)
% Define number of segments and iterations and get the output. 
% Org is image. You can use your image. 
% ParVal.N is Number of Segments
% ParVal.MaxIt is Number of runs
%----------------------------------------------------------------------
clc;
clear;
close all;

%% Load Image
Org=imread('Veg.jpg');
X = rgb2gray(Org);
X=double(X);
img=X;
X=X(:)';

%% Neural Gas Network (NGN) Parameters

ParVal.N = 16; % Number of Segments
ParVal.MaxIt = 50; % Number of runs

ParVal.tmax = 100000;

ParVal.epsilon_initial = 0.3;
ParVal.epsilon_final = 0.02;
ParVal.lambda_initial = 2;
ParVal.lambda_final = 0.1;
ParVal.T_initial = 5;
ParVal.T_final = 10;

%% Training Neural Gas Network
NGNnetwok = GasNN(X, ParVal);

%% Vector to image and plot
Weight=sum(round(rescale(NGNnetwok.w,1,ParVal.N)));
Weight=round(rescale(Weight,1,ParVal.N));
indexed=reshape(Weight(1,:),size(img));
segmented = label2rgb(indexed); 
% Plot Res
figure('units','normalized','outerposition',[0 0 1 1])
subplot(2,2,1)
imshow(Org,[]); title('Original');
subplot(2,2,2)
imshow(img,[]); title('Grey');
subplot(2,2,3)
imshow(segmented);
title(['Segmented in [' num2str(ParVal.N) '] Segments']);
subplot(2,2,4)
imshow(indexed,[]);
title(['Quantized in [' num2str(ParVal.N) '] Thresholds']);


