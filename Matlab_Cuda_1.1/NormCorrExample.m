clear all;
clear all;
close all;
clc;


in_file = 'red_cell.avi';
figure;
Nparticles = 100;
speed = .01;
Fxy = cell(size(in_file));
J = avireadgray(in_file,1);
info = aviinfo(in_file);
Isz = [info.Height info.Width];
Nfr = info.NumFrames;


imshow(J,[],'init',400);
text(0,0,'crop the cell of interest, when done double click', 'BackgroundColor',[.7 .9 .7]);
title(in_file);
[template rect] = imcrop;
x0 = round(rect(1)+rect(3)/2);
y0 = round(rect(2)+rect(4)/2);

% [x_pf, y_pf] = avi_PF(in_file, template, Nparticles, x0, y0, speed);

% set initial position
x_pf = x0; 
y_pf = y0;

% initial likelihood
likelihood = zeros(Nparticles,1);

% initial particles
x = x0*ones(Nparticles,1);
y = y0*ones(Nparticles,1);
F = zeros(Isz(1), Isz(2)); 

template = double(template);

temp = aviread(in_file,1);
S = size(temp.cdata);

figure(1)
hI = imshow(J,[],'init',400);
hold on
hp1 = plot(NaN,NaN,'.r');
hxe = plot(NaN,NaN,'.b-');
hold off    
tsize = size(template); 
for k = 2:Nfr
    Ik = double(avireadgray(in_file,k));
    % apply motion model
    % draw samples from motion model 
    % moves to the right in x
    % doesn't move much in y
    x = x + speed +(2*randn(Nparticles,1));
    y = y + 2*randn(Nparticles,1);

    %in case particles end up outside of the image
    x(x > size(Ik,2)) = size(Ik,2);
    y(y > size(Ik,1)) = size(Ik,1);
    x(x < 1) = 1;
    y(y < 1) = 1;
        
    %%%%%%%%%%% This whole thing can be the function %%%%%%%%%%%%%
    
    likelihood = GetLikelihood(Ik,template,x,y);
    %%%%%%%%%%%%%%%%%%%%%% end of function %%%%%%%%%%%%%%%%%%%%%
    
    % update & normalize weights
    weights = likelihood/sum(likelihood);    

    % estimate the object location by expected values
    x_pf(k) = sum(x.*weights);
    y_pf(k) = sum(y.*weights);
    
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
    set(hI,'Cdata',Ik);
    title(k);
    set(hp1, 'Xdata', x, 'Ydata',y);
    set(hxe, 'Xdata', x_pf(1:k), 'Ydata', y_pf(1:k));
%     text(x,y,num2str(weights,'%0.4f'),'Parent',haxes(1));
    pause(0.01)
    % reset likelihood image
    F = zeros(Isz(1), Isz(2));
    
    % resampled x and y
    x = xj;
    y = yj;  

end
