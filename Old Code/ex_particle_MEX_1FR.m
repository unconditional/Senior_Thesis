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
Nfr = 3;
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
    
Nparticles = 100000;

posX = x0;
posY = y0;
disp('POSX');
disp(posX);
disp('POSY');
disp(posY); 
for k = 2:Nfr
	% current image
    Ik = uint8(I(:,:,k)); 
	[posX,posY] = ex_particle_OPENMP(Ik, Isz(1), Isz(2), Nparticles, posX, posY);
	disp('X');
	disp(posX);
	disp('Y');
	disp(posY);
end
