function [imSeg, Psi] = level_set_otsu(imgo, dt, N, showRes)

tic
if nargin == 0
    close all
    
    % Underlying image
    imgo = -(2*petals(256,8,100)-1);
    
    % Reference image (for result comparison)
    imRef = imgo < 0;
    
    % Add noise to the image
    imgo = double(imnoise(imgo,'salt & pepper',.5));
    
    % Num. of iterations and step
    N = 200; dt = .1;
    
    % Show results
    showRes = 1;
end

% Define Initial Front
Lo = double(createCirclesLevelSet(size(imgo)));

% Define Morphological Parameters
SEg = strel('disk',3); % ou 1
SEgMorfStep = strel('disk',1); % ou 1

% Define MedianDisc Parameters
Nb=[0 1 0; 1 1 1;0 1 0];
med=3;

% Inicialization of Surface and others
Psi = -((2*Lo)-1);
Psi_d = imdilate(Psi,SEg);
Psi_e = imerode(Psi,SEg);

%% Main procedure
c = 1; F = 0;
while (c<N)%&&((sum(thr<100))<40) %Convergence
    c = c+1;
    
    % Front Propagation
    Psi = ordfilt2(Psi + dt*((max(F,0).*Psi_d - min(F,0).*Psi_e)),med,Nb);
    Psi_d = imdilate(Psi,SEg);
    Psi_e = imerode(Psi,SEg);
    
    % Morphological Regularization
    Psi = imdilate(imerode(Psi,SEgMorfStep),SEgMorfStep);
    Psi = imerode(imdilate(Psi,SEgMorfStep),SEgMorfStep);
    
    % Front update
    F = newFront(imgo, Psi < 0);    
    
    if(showRes)
        imshow(Psi > 0,[]);pause(0.01);
    end
end

% Segment image
thres = 0;
imSeg = Psi < thres;

% Show results
if(showRes)
    showFinalResults(imgo, imSeg);
    disp(error_seg(imSeg,imRef));
    toc
end

end

function newF = newFront(I, D)
% Apply otsu's front update
opD = ~D;
newF = I.^2 ...
    - 2*mean(mean(I(D))).*I - 2*mean(mean(I(opD))).*I ...
    + mean(mean(I(D))).^2 + mean(mean(I(opD))).^2;
end

function showFinalResults(imgo, imSeg)
    BW = imdilate(edge(imSeg),strel('disk',1));
    
    imgo = imgo/max(max(imgo));
    imgo(:,:,1) = imgo.*~BW;
    imgo(:,:,2) = imgo(:,:,1);
    imgo(:,:,3) = imgo(:,:,1);
    imgo(:,:,1) = min(imgo(:,:,1) + BW, 1);
    
    figure
    subplot(1,2,1)
        imshow(imgo,[]);
    subplot(1,2,2)
        imshow(imSeg,[])
        title('Zero Level Set')    
    
end

function img = petals(size, N, R)
    X = size;
    img = zeros(X);
    X = X/2;
    R = R*X/size;

    p = N;
    theta = -pi/2:pi/(p*360):pi/2;
    r = 0.95*cos(p.*theta);
    r = r(1+180*(p-1):361+180*(p-1));
    theta = theta(1+180*(p-1):361+180*(p-1));

    phi = 2*pi/N;
    for n = 0:N-1
        x = round(1+X*(1+r.*cos(n*phi+theta)));
        y = round(1+X*(1+r.*sin(n*phi+theta)));
        for i = 1:length(x)
            img(y(i),x(i))=1;
        end
    end

    theta = 0:2*pi/(R*360):2*pi;
    x = 1+X+round(R*cos(theta));
    y = 1+X+round(R*sin(theta));
    for i = 1:length(x)
        img(y(i),x(i)) = 1;
    end

    img = imfill(img,'holes');
end

function [Lo] = createCirclesLevelSet(sizeImg)
    if nargin == 0
        sizeImg = [100 100];
    end

    phi = imread('level_zero.bmp');
    Lo = phi(1:sizeImg(1),1:sizeImg(2));
end

function err = error_seg(imSeg, imRef, flag)
    if(~exist('flag','var'))
        flag = 1;
    end

    if(flag == 1)
        e1 = error_seg(imSeg, imRef, 0);
        e2 = error_seg(~imSeg, imRef, 0);

        if(e1.Dice > e2.Dice)
            err = e1;
        else
            err = e2;
        end
    else
        imRef = imRef(:); imSeg = imSeg(:);
        common = sum(imRef & imSeg);
        union = sum(imRef | imSeg);
        cm = sum(imRef); 
        co = sum(imSeg);
        
        err.Jaccard = common/union;
        err.Dice = (2 * common)/(cm+co);
        err.RFP = (co-common)/cm;
        err.RFN = (cm-common)/cm;

        R1 = (imSeg==0) .* (imRef==1);
        R2 = (imSeg==0) .* (imRef==0);
        E1 = sum(R1(:)) - sum(R2(:));

        R3 = (imSeg==1) & (imRef==1) .* sign(sign(E1)+1);
        R4 = (imSeg==1) & (imRef==0) .* sign(sign(-E1)+1);
        R1 = R1 .* sign(sign(-E1)+1);
        R2 = R2 .* sign(sign(E1)+1);

        EoS = sum(R1(:)) + sum(R2(:)) + sum(R3(:)) + sum(R4(:));
        if ~isempty(find(isnan(imSeg), 1))
            EoS = numel(imRef);
        end

        err.EoS = EoS/(length(imRef(:)));
    end
end
