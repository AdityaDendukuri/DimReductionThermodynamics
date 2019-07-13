    function [coords, L] = initCubicGrid(nPart,density)
    
%     nlat=((nPart)^(1/3))/4
%     bx=5;
%     by=5;
%     dlatx=bx/nlat;
%     dlaty=by/nlat;
%     ipart=0
%     for ix=1,nlat
%         for iy=1,nlat
%             ipart=ipart+1
%             x(1,ipart)=ix*dlatx
%             x(2,ipart)=iy*dlaty
%         end
%     end
    
            
        % Initialize with zeroes
        coords = zeros(3,nPart);
    
        % Get the cooresponding box size
        L = (nPart/density)^(1.0/3.0);
    
        % Find the lowest perfect cube greater than or equal to the number of
        % particles
        nCube = (nPart)^(1/3)
%         sqrt(nPart)+0.1;
        
        while (nCube^3 < nPart)
            nCube = nCube + 1;
        end
        
        
        % Start positioning - use a 3D index for counting the spots
        index = [0,0,0]';
        
        % Assign particle positions
        for part=1:nPart
            % Set coordinate
            coords(:,part) = ((index+[0.5,0.5,0.5]')*(L/nCube));
            
            % Advance the index
            index(1) = index(1) + 1;
            if (index(1) == nCube) 
                index(1) = 0;
                index(2) = index(2) + 1;
                if (index(2) == nCube)
                    index(2) = 0;
                    index(3) = index(3) + 1;
                    if (index(3) == nCube)
                        index(3) = 0
                end
            end
        end
    
    end