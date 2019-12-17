% The symmetrized Itoh–Abe DRG method for Example 2 in
% "Energy preserving methods on Riemannian manifolds" by Celledoni, Eidnes,
% Owren and Ringholm (Mathematics of Computation, 2019)
%
% Code written by Sølve Eidnes
%
close all
g = 9.81;
a = 1;
b = 2;
d = 2;
H = @(qp) .5*qp(4:6)'*qp(4:6) + g*qp(3);
DH = @(qp) [0;0;g;qp(4:6)];
S = [zeros(3),eye(3);-eye(3),zeros(3)];
% Initalize the start values:
qp = zeros(6,1);
qp(1) = .5; qp(2) = .5; qp(3) = .5*(qp(1)^2/a^2+qp(2)^2/b^2-d);
qp(4) = 1.5*1; qp(5) = 3.5*1; qp(6) = qp(1)*qp(4)/a^2+qp(2)*qp(5)/b^2;
qpstart = qp;
%
options = optimset('Display','off','TolFun',1e-14);
c = @(qp,qpn) conm(qp,qpn,a,b,d);

h = .01;
M = 2000;
qv = [qp(1:3),zeros(3,M)];
pv = [qp(4:6),zeros(3,M)];
Hs1 = [H(qp);zeros(M,1)];
W = @(qp,qpn,h) phibinv(c(qp,qpn),qp,a,b,d) + h*.5*S*(gradH_IAc(qp,qpn,c(qp,qpn),a,b,d)+gradH_IAc(qpn,qp,c(qp,qpn),a,b,d));
for i = 1:M
    F = @(qpn) qpn - phib(c(qp,qpn),W(qp,qpn,h),a,b,d);
    qps = initguess(qp,a,b,d,h);
    qp = fsolve(F,qps,options);
    qv(:,i+1) = qp(1:3);
    pv(:,i+1) = qp(4:6);
    Hs1(i+1) = H(qp);
end

figure(3)
[x,y,z]= meshgrid(-2*a:0.025:2*a,-2*b:0.025:2*b,-d/2:0.025:0);
zofxy = @(x,y) .5*(x.^2/a^2 + y.^2/b^2 - d); % Elliptic paraboloid
p = @(x,y,z) z - zofxy(x,y);
v = p(x,y,z);
p = patch(isosurface(x,y,z,v,0));
isonormals(x,y,z,v,p)
p.FaceColor = 'yellow';
p.EdgeColor = 'none';
p.FaceAlpha = 1;
p.EdgeAlpha = 0;
p.LineWidth = 0.1;
daspect([1 1 1])
view([145 25])
axis tight
hold on
cdata = z;
grid on
isocolors(x,y,z,cdata,p)
p.FaceColor = 'flat';
p.EdgeColor = 'none';
colormap default
plot3(qv(1,:),qv(2,:),qv(3,:),'k','linewidth',0.5)

function out = phib(qp,xy,a,b,d) % Retraction for the cotangent bundle
    q = qp(1:3);
    p = qp(4:6);
    x = xy(1:3);
    y = xy(4:6);
    alpha = sqrt((q(3)+x(3))^2+d*((q(1)+x(1))^2/a^2+(q(2)+x(2))^2/b^2));
    qn = d/(alpha-q(3)-x(3))*(q+x);
    pn = p + y - (qn(1)/a^2*(p(1)+y(1))+qn(2)/b^2*(p(2)+y(2))-p(3)-y(3))/(qn(1)^2/a^4+qn(2)^2/b^4+1)*[qn(1)/a^2;qn(2)/b^2;-1];
    out = [qn;pn];
end

function out = phibinv(qp,qpn,a,b,d) % Inverse retraction for the cotangent bundle
    q = qp(1:3);
    p = qp(4:6);
    qn = qpn(1:3);
    pn = qpn(4:6);
    x = (q(3)+d)/(qn(1)*q(1)/a^2+qn(2)*q(2)/b^2-qn(3))*qn-q;
    alpha = -(1/a^2*q(1)*(pn(1)-p(1))+1/b^2*q(2)*(pn(2)-p(2))-pn(3)+p(3))/(q(1)*qn(1)/a^4+q(2)*qn(2)/b^4+1);
    y = pn-p+alpha*[qn(1)/a^2;qn(2)/b^2;-1];
    out = [x;y];
end

function c = conm(qp,qpn,a,b,d) % center point c on the manifold
    A = (qpn(1)+qp(1))^2/a^2+(qpn(2)+qp(2))^2/b^2;
    B = -2*(qpn(3)+qp(3));
    if A == 0
        c = [0;0;0;0;0;0];
    else
        alpha = (-B+sqrt(B^2+4*A*d))/(2*A);
        c = [alpha*(qp(1:3)+qpn(1:3));.5*(qp(4:6)+qpn(4:6))];
    end
    c = [c(1:5);c(1)*c(4)/a^2+c(2)*c(5)/b^2];
end

function qpn = initguess(qp,a,b,d,h) % Initial guess from Forward Euler
    g = 9.81;
    RGH = @(qp) [0;0;g;qp(4:6)] - 1/(qp(1)^2/a^4+qp(2)^2/b^4+1)*[-g*qp(1)/a^2;-g*qp(2)/b^2;g;
        (qp(4)*qp(1)/a^2+qp(2)*qp(5)/b^2-qp(6))*qp(1)/a^2;(qp(4)*qp(1)/a^2+qp(2)*qp(5)/b^2-qp(6))*qp(2)/b^2;-(qp(4)*qp(1)/a^2+qp(2)*qp(5)/b^2-qp(6))];
    S = [zeros(3),eye(3);-eye(3),zeros(3)];
    qpn = phib(qp,h*S*RGH(qp),a,b,d);
end

function gradH = gradH_IAc(qp,qpn,c,a,b,d) % Itoh-Abe DRG
    g = 9.81;
    n = [c(1)/a^2,c(2)/b^2,-1];
    nn = null(n);
    E = [nn,zeros(3,2);
        zeros(3,2),nn];
    H = @(qp) .5*qp(4:6)'*qp(4:6) + g*qp(3);
    RGH = @(qp) [0;0;g;qp(4:6)] - 1/(qp(1)^2/a^4+qp(2)^2/b^4+1)*[-g*qp(1)/a^2;-g*qp(2)/b^2;g;
        (qp(4)*qp(1)/a^2+qp(2)*qp(5)/b^2-qp(6))*qp(1)/a^2;(qp(4)*qp(1)/a^2+qp(2)*qp(5)/b^2-qp(6))*qp(2)/b^2;-(qp(4)*qp(1)/a^2+qp(2)*qp(5)/b^2-qp(6))];
    gradH = 0;
    eta = phibinv(c,qp,a,b,d);
    w = qp;
    diffa = phibinv(c,qpn,a,b,d)-phibinv(c,qp,a,b,d);
    alpha = [diffa'*E(:,1); diffa'*E(:,2);
        diffa'*E(:,3); diffa'*E(:,4)];
    for j = 1:4
        eta = eta + alpha(j)*E(:,j);
        wj = phib(c,eta,a,b,d);
        if alpha(j) == 0
            gradH = gradH + RGH(w).*E(:,j);
        else
            gradH = gradH + (H(wj)-H(w))/alpha(j)*E(:,j);
        end
        w = wj;
    end
end