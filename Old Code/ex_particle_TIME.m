% A simple implementation of SIS paprticel filter (Algorithm 1 of
% Arulampalam et al. Tutorial
clear all, close all, pause(0)
clc;

%% VIDEO SEQUENCE

% the synthetic video sequence we will work with here is composed of a
% single moving object, circular in shape (fixed radius)
% The motion here is a linear motion
% the foreground intensity and the backgrounf intensity is known
% the image is corrupted with zero mean Gaussian noise
start = tic;
Isz = [128 128];
Nfr = 10;
I = zeros([Isz,Nfr]);

% object center
x0 = round(Isz(2)/2); 
y0 = round(Isz(1)/2);
I(y0,x0) = 1;

% move point
for k = 2:Nfr
    xk = x0 + (k-1);
    yk = y0 - 2*(k-1);
	if(xk <= 0)
		xk = 1;
	end
	if(yk <= 0)
		yk = 1;
	end
    I(yk,xk,k) = 1;
end

% create object
se = strel('disk',5);
I = imdilate(I,se);

% define background, add noise
I(I==0) = 100;
I(I==1) = 228;
I = uint8(I + 5*randn(size(I)));

vEnd = toc(start);
disp('VIDEO SEQUENCE TOOK');
disp(vEnd);
disp('-------------------');
pStart = tic;
%% PARTICLE FILTER
% For the particle filter pretend that you don't know the motion model
% so assume a random walk model. This is in general not a good idea.
% If you have some information about the motion of the object, try to
% create a motion model and work with it
    
Nparticles = 1000;

% original particle centroid
xe = x0;
ye = y0;

% expected object locations, compared to center
objxy = getneighbors(se);

% initial weights are all equal
weights = (1/Nparticles)*ones(Nparticles,1);

% initial likelihood
likelihood = zeros(Nparticles,1);

x = x0*ones(Nparticles,1);
y = y0*ones(Nparticles,1);

for k = 2:Nfr
    
    % current image
    Ik = double(I(:,:,k));    
	applyMotionModel = tic;
    % apply motion model
    % draw samples from motion model (random walk). The only prior information 
    % is that the object moves twice as fast in the y direction
    x = x + 1 + 5*randn(Nparticles,1);
    y = y - 2 + 2*randn(Nparticles,1);
    applyMotionModelEnd = toc(applyMotionModel);
	disp('APPLYING MOTION MODEL TOOK');
	disp(applyMotionModelEnd);
	disp('--------------------------');
	filterLikelihoods = tic;
    % particle filter likelihoods
    for np = 1:Nparticles
                
        % compute the likelihood: remember our assumption is that you know
        % foreground and the background image intensity distribution.
        % Notice that we consider here a likelihood ratio, instead of
        % p(z|x). It is possible in this case. why? a hometask for you.
        ind = sub2ind(Isz,round(y(np))+objxy(:,2),round(x(np))+objxy(:,1));
        likelihood(np) = sum( ( (Ik(ind)-100).^2 - (Ik(ind)-228).^2 ) / (2*5^2) );
        likelihood(np) = likelihood(np)/length(ind);
    end
    filterLikelihoodsEnd = toc(filterLikelihoods);
	disp('FILTER LIKELIHOODS TOOK');
	disp(filterLikelihoodsEnd);
	disp('--------------------------');
	expBegin = tic;
    % update & normalize weights
    % using equation (63) of Arulampalam Tutorial
    weights = weights .* exp(likelihood);
	 expEnd = toc(expBegin);
	disp('EXPONENTIALS TOOK');
	disp(expEnd);
	disp('--------------------------');
	sumWeights = tic;
    weights = weights/sum(weights);
	sumWeightsEnd = toc(sumWeights);
	disp('SUM & NORMALIZE WEIGHTS TOOK');
	disp(sumWeightsEnd);
	disp('--------------------------');
	
	moveObjBegin = tic;
    % estimate the object location by expected values
    xe = sum(x.*weights);
    ye = sum(y.*weights);
	disp('XE:');
	disp(xe);
	disp('--------------------------');
	disp('YE:');
	disp(ye);
	disp('--------------------------');
    difference = sqrt((xe-x0)^2 + (ye-y0)^2);
    disp(difference);
    moveObjEnd = toc(moveObjBegin);
	disp('MOVING OBJ TOOK');
	disp(moveObjEnd);
	disp('--------------------------');
    
    % display
    %figure(1)
    %imshow(I(:,:,k),'init','fit')
    %hold on
    %plot(x,y,'.r')
    %plot(xe,ye,'*'); 
    %hold off
    
    %pause(0.5);
    
    % resampling
	calcCDF = tic;
	CDF = cumsum(weights);
	calcCDFEnd = toc(calcCDF);
	disp('CALC CDF TOOK');
	disp(calcCDFEnd);
	disp('--------------------------');
	calcU = tic;
    xj = zeros(Nparticles,1);
    yj = zeros(Nparticles,1);
    
    u1 = (1/Nparticles)*rand(1);
    u  = u1 + (0:Nparticles-1)/Nparticles;
	calcUEnd = toc(calcU);
	disp('CALC U TOOK');
	disp(calcUEnd);
	disp('--------------------------');
	update = tic;
    for j = 1:Nparticles
        i = find(CDF > u(j),1);
        xj(j) = x(i);
        yj(j) = y(i);
    end   
    updateEnd = toc(update);
	disp('UPDATING TOOK');
	disp(updateEnd);
	disp('--------------------------');
	reset = tic;
    x = xj;
    y = yj;   
    weights(:) = 1/Nparticles;
	resetEnd = toc(reset);
	disp('RESETING TOOK');
	disp(resetEnd);
	disp('--------------------------');
end
pEnd = toc(pStart);
disp('PARTICLE FILTER TOOK:');
disp(pEnd);
disp('PROGRAM TOOK:');
disp(pEnd + vEnd);
