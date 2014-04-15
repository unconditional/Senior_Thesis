function likelihood = GetLikelihood(I,template,x,y)

% get window with particles
    xlb = min(x(:))-size(template,2);
    xlb(xlb<1) = 1;
    ylb = min(y(:))-size(template,1);
    ylb(ylb<1) = 1;
    
    xub = max(x(:))+size(template,2);
    xub(xub>size(I,2)) = size(I,2);
    yub = max(y(:))+size(template,1);
    yub(yub>size(I,1)) = size(I,1);
    
    
    rect = round([xlb,ylb,xub-xlb,yub-ylb]);
%         max(x(:))-min(x(:))+2*size(template,2),max(y(:))-min(y(:))+2*size(template,1)]);
    window = I(rect(2):rect(2)+rect(4)-1,rect(1):rect(1)+rect(3)-1);
    match = normxcorr2_mex(double(template),double(window),'same');

    match(match<0) =0; % saturate because not looking for inversly correlated
    F(rect(2):rect(2)+rect(4)-1,rect(1):rect(1)+rect(3)-1) = match;
 
    % particle filter likelihoods
    ind = sub2ind(size(F),round(y),round(x));
    likelihood = F(ind);    
    % if target is comletely lost
    if sum(likelihood)==0, likelihood(:) = 1; end