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
ex_particle_OPENMP_full(avireadgray(in_file), Isz(1), Isz(2), Nfr, Nparticles, x0, y0, template);