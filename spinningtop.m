% An implementation of Example 1 from
% "Energy preserving methods on Riemannian manifolds" by Celledoni, Eidnes,
% Owren and Ringholm (Mathematics of Computation, 2019)
%
% Code written by Sølve Eidnes and Torbjørn Ringholm 
%
close all
tic
mn = @(p,q) .5*sqrt(sum((p+q).^2)); % 2-norm of m
phi = @(p,v) .5*(p+v)/mn(p,v); % Retraction
phiinv = @(p,q) q/(p'*q)-p; % Inverse retraction
Inrinv = diag([1 .5 .25]); % Moments of inertia
H = @(p) .5*(p+2/3*p.^2)'*(Inrinv*p); % Energy
N = 300; % Number of time steps
k = .1; % Time step size
x = [1 1 1]'/norm([1 1 1]);
xv = [x,zeros(3,N)];
Hv = [H(x),zeros(1,N)];
options = optimset('Display','off','TolFun',1e-16);
method = 2;
if method == 1 % AVF DRGM
    c = @(p,q) .5*(p+q)/mn(p,q);
    W = @(x,xn,k) phiinv(c(x,xn),x) + k*cross(.5*(x+xn),ddH_AVF(x,xn,c(x,xn)));
    for i = 1:N
        F = @(xn) xn - phi(c(x,xn),W(x,xn,k));
        x = fsolve(F,x,options);
        Hv(i+1) = H(x);
        xv(:,i+1) = x;
    end
elseif method == 2 % Gonzalez' midpoint DRGM
    c = @(p,q) .5*(p+q)/mn(p,q); 
    W = @(x,xn,k) phiinv(c(x,xn),x) + k*cross(.5*(x+xn),ddH_MP(x,xn));
    for i = 1:N
        F = @(xn) xn - phi(c(x,xn),W(x,xn,k)); 
        x = fsolve(F,x,options);
        Hv(i+1) = H(x);
        xv(:,i+1) = x;
    end
elseif method == 3 % Itoh-Abe DRGM
    W = @(x,xn,k) k*cross(.5*(x+xn),ddH_CI_c(x,xn)); 
    for i = 1:N
        F = @(xn) xn - phi(x,W(x,xn,k));
        xs = x + k*cross(x,DH(x));
        x = fsolve(F,xs,options);
        Hv(i+1) = H(x);
        xv(:,i+1) = x;
    end
elseif method == 4 % Composition Itoh-Abe DRGM of order 2, c midpoint
    c = @(p,q) .5*(p+q)/mn(p,q); 
    W1 = @(x,xn,k) phiinv(c(x,xn),x) + .5*k*cross(.5*(x+xn),ddH_CI(xn,x,c(x,xn)));
    W2 = @(x,xn,k) phiinv(c(x,xn),x) + .5*k*cross(.5*(x+xn),ddH_CI(x,xn,c(x,xn)));
    DH = @(p) Inrinv*(p+p.^2);
    for i = 1:N
        F = @(xn) xn - phi(c(x,xn),W1(x,xn,k));
        xs = x + .5*k*cross(x,DH(x));
        xh = fsolve(F,xs,options);
        F = @(xn) xn - phi(c(xh,xn),W2(xh,xn,k));
        xs = xh + .5*k*cross(xh,DH(xh));
        x = fsolve(F,xs,options);
        Hv(i+1) = H(x);
        xv(:,i+1) = x;
    end
elseif method == 5 % Symmetrized Itho-Abe DRGM
    c = @(p,q) .5*(p+q)/mn(p,q); 
    W = @(x,xn,k) phiinv(c(x,xn),x) + k*.5*cross(.5*(x+xn),ddH_CI(x,xn,c(x,xn))+ddH_CI(xn,x,c(x,xn)));
    DH = @(p) Inrinv*(p+p.^2);
    for i = 1:N
        F = @(xn) xn - phi(c(x,xn),W(x,xn,k));
        xs = x + k*cross(x,DH(x)); xs = xs/norm(xs);
        x = fsolve(F,xs,options);
        Hv(i+1) = H(x);
        xv(:,i+1) = x;
    end
end
toc

figure(1)
[X,Y,Z]=sphere(100);
r=1;
surf(r*X,r*Y,r*Z,'LineStyle','None','FaceAlpha',0.6)
hold on
plot3(xv(1,:),xv(2,:),xv(3,:),'k','LineWidth',2)
view([135 30])
hold off
figure(2)
plot(k*(0:N),Hv-Hv(1),'LineWidth',2)

function dH = ddH_AVF(p,q,c) % AVF DRG
    phiinv = @(c,q) q/(c'*q)-c; % Inverse retraction
    Inrinv = diag([1 .5 .25]);
    pip = phiinv(c,p);
    piq = phiinv(c,q);
    l = @(xi) c + (1-xi)*pip + xi*piq;
    integrand = @(xi) 1/norm(l(xi))^2*(Inrinv*(l(xi)+l(xi).^2/norm(l(xi))) - 1/norm(l(xi))^2*(l(xi)'*(Inrinv*(l(xi)+l(xi).^2/norm(l(xi)))))*l(xi));
    dH = integral(integrand,0,1,'RelTol',1e-12,'AbsTol',1e-18,'ArrayValued',true);
end

function dH = ddH_MP(p,q) % Midpoint (Gonzalez) DRG
    mn = .5*sqrt(sum((p+q).^2));
    Inrinv = diag([1 .5 .25]);
    H = @(p) .5*(p+2/3*p.^2)'*(Inrinv*p);
    if p == q
        dH = 1/mn*(Inrinv*(1/3*(p.^2+p.*q+q.^2)+.5*(p+q)));
    else
        dH = 1/mn*(Inrinv*(1/3*(p.^2+p.*q+q.^2)+.5*(p+q)) + (mn^2-1)/sum((q-p).^2)*(H(q)-H(p))*(q-p));
    end
end

function dH = ddH_CI_c(p,q) % Itoh-Abe DRG w/ c = p
    E = null(p');
    Inrinv = diag([1 .5 .25]);
    phi = @(p,v) (p+v)/sqrt(sum((p+v).^2));
    phiinv = @(p,q) q/(p'*q)-p; 
    H = @(p) .5*(p+2/3*p.^2)'*(Inrinv*p);
    DH = @(p) Inrinv*(p+p.^2);
    dH = 0;
    w = p;
    alpha = [phiinv(p,q)'*E(:,1); phiinv(p,q)'*E(:,2)];
    for j = 1:2
        eta = phiinv(p,w) + alpha(j)*E(:,j);
        wj = phi(p,eta);
        if alpha(j) == 0
            dH = dH + DH(w).*E(:,j);
        else
            dH = dH + (H(wj)-H(w))/alpha(j)*E(:,j);
        end
        w = wj;
    end
end

function dH = ddH_CI(p,q,c) % Itoh-Abe DRG
    E = null(c');
    Inrinv = diag([1 .5 .25]);
    phi = @(p,v) (p+v)/sqrt(sum((p+v).^2));
    phiinv = @(p,q) q/(p'*q)-p;
    H = @(p) .5*(p+2/3*p.^2)'*(Inrinv*p);
    DH = @(p) Inrinv*(p+p.^2);
    dH = 0;
    w = p;
    diffa = phiinv(c,q)-phiinv(c,p);
    alpha = [diffa'*E(:,1); diffa'*E(:,2)];
    for j = 1:2
        eta = phiinv(c,w) + alpha(j)*E(:,j);
        wj = phi(c,eta);
        if alpha(j) == 0
            dH = dH + DH(w).*E(:,j);
        else
            dH = dH + (H(wj)-H(w))/alpha(j)*E(:,j);
        end
        w = wj;
    end
end