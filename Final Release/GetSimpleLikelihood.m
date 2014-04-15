function likelihood = GetSimpleLikelihood(Ik, objxy, x, y)
	Nparticles = size(x);
	Nparticles = Nparticles(2);
	countOnes = size(objxy);
	countOnes = countOnes(1);
	% particle filter likelihoods
    for np = 1:Nparticles
                
        % compute the likelihood: remember our assumption is that you know
        % foreground and the background image intensity distribution.
        % Notice that we consider here a likelihood ratio, instead of
        % p(z|x). It is possible in this case. why? a hometask for you.
        ind = sub2ind(Ik,round(y(np))+objxy(:,2),round(x(np))+objxy(:,1));
        likelihood(np) = sum( ( (Ik(ind)-100).^2 - (Ik(ind)-228).^2 ) / (2*5^2) );
        likelihood(np) = likelihood(np)/length(ind);
    end
end