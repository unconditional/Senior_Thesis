function movgray = avireadgray(filename,varargin)

%AVIREADGRAY return grayscale frames of an AVI sequence. 
%
%   We use the function AVIREAD to read the image, and convert it to 
%   grayscale using RGB2GRAY or IND2GRAY.  Inputs are the same as 
%   AVIREAD; output is a HEIGHT x WIDTH x length(INDEX) array.
%
%   Note, we deterimine AVI type using the "ImageType" property.
%   Also, the two types come in the following forms:
%
%       Image Type      cdata Field     colormap Field
%       Truecolor       h x w x 3       Empty
%       Indexed         h x w           m x 3 

%WRITTEN BY:        DREW GILLIAM
%MODIFIED BY:       DREW GILLIAM
%LAST MODIFIED:     2005.10



%==========================================================================
% READ AVI
%==========================================================================

% avi info (includes filename check)
info = aviinfo(filename);
M    = info.Height;
N    = info.Width;
type = info.ImageType;

% read avi data
mov = aviread(filename,varargin{:});
NF  = length(mov);


%==========================================================================
% CONVERT TO GRAYSCALE IMAGE
% AVIs come in two distinct types: truecolor (data only) and indexed 
% (data + colormap). We convert both to grayscale using the 
% appropriate function. 
%==========================================================================

% initialize gray mov, same class as input
movgray = zeros(M,N,NF,class(mov(1).cdata));

switch lower(type)
    
    % convert truecolor
    case 'truecolor'
        for i = 1:NF
            movgray(:,:,i)= ...
                rgb2gray(mov(i).cdata);
        end
        
    % convert indexed    
    case 'indexed'
        for i = 1:NF
            movgray(:,:,i) = ...
                ind2gray(mov(i).cdata, mov(i).colormap);
        end
        
    % unknown image type
    otherwise
        error('ImageType not recognized.');
        
end

% remove original mov from memory
clear mov;

return



%**************************************************************************
% END OF FILE
%**************************************************************************