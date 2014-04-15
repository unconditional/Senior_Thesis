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

Isz = [128 128];
Nfr = 20;
I = zeros([Isz,Nfr]);

% object center
x0 = round(Isz(2)/2); 
y0 = round(Isz(1)/2);
I(y0,x0) = 1;

% move point
for k = 2:Nfr
    xk = x0 + (k-1);
    yk = y0 - 2*(k-1);
    I(yk,xk,k) = 1;
end

% create object
se = strel('disk',5);
I = imdilate(I,se);

% define background, add noise
I(I==0) = 100;
I(I==1) = 228;
I = uint8(I + 5*randn(size(I)));


%% PARTICLE FILTER
% For the particle filter pretend that you don't know the motion model
% so assume a random walk model. This is in general not a good idea.
% If you have some information about the motion of the object, try to
% create a motion model and work with it
    
Nparticles = 100;

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

    % apply motion model
    % draw samples from motion model (random walk). The only prior information 
    % is that the object moves twice as fast in the y direction
    x = x + 1 + 5*randn(Nparticles,1);
    y = y - 2 + 2*randn(Nparticles,1);
    
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
    
    % update & normalize weights
    % using equation (63) of Arulampalam Tutorial
    weights = weights .* exp(likelihood);
    weights = weights/sum(weights);
    
    % estimate the object location by expected values
    xe = sum(x.*weights);
    ye = sum(y.*weights);
    difference = sqrt((xe-x0)^2 + (ye-y0)^2);
    disp(difference);
    
    
    % display
    figure(1)
    imshow(I(:,:,k),'init','fit')
    hold on
    plot(x,y,'.r')
    plot(xe,ye,'*'); 
    hold off
    
    pause(0.5);
    
    % resampling
    xj = zeros(Nparticles,1);
    yj = zeros(Nparticles,1);
    
    CDF = cumsum(weights);
    u1 = (1/Nparticles)*rand(1);
    u  = u1 + (0:Nparticles-1)/Nparticles;
    for j = 1:Nparticles
        i = find(CDF > u(j),1);
        xj(j) = x(i);
        yj(j) = y(i);
    end   
    
    x = xj;
    y = yj;   
    weights(:) = 1/Nparticles;
end

