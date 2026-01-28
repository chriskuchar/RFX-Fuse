#include "rf_arrays.hpp"

namespace rf {

// Global arrays instance
RFArrays g_arrays;

void RFArrays::allocate(integer_t ns, integer_t nt, integer_t md, integer_t n_classes, integer_t mc, integer_t mn) {
    // Set global parameters
    nsample = ns;
    ntree = nt;
    mdim = md;
    nclass = n_classes;
    maxcat = mc;
    maxnode = mn;
    mimp = imp * (mdim - 1) + 1;

    // Main data arrays
    x.resize(nsample * mdim);           // (nsample, mdim) row-major
    cl.resize(nsample);
    weight.resize(nsample);
    win.resize(nsample);
    v.resize(nsample);
    q.resize(nsample * nclass);         // (nsample, nclass) row-major
    qimp.resize(nsample);
    qimpm.resize(nsample * mdim);       // (nsample, mdim) row-major
    wr.resize(nclass);
    wl.resize(nclass);
    tmpclass.resize(nclass);
    tclasscat.resize(nclass * maxcat);  // (nclass, maxcat)
    avimp.resize(mdim);
    sqsd.resize(mdim);
    xbestsplit.resize(maxnode);
    tnodewt.resize(maxnode);
    classpop.resize(nclass * maxnode);  // (nclass, maxnode)
    tclasspop.resize(nclass);

    // Index and working arrays
    asave.resize(mdim * nsample);       // (mdim, nsample)
    amat.resize(mdim * nsample);        // (mdim, nsample)
    ties.resize(mdim * nsample);        // (mdim, nsample)
    nout.resize(nsample);
    nin.resize(nsample);
    joob.resize(nsample);
    pjoob.resize(nsample);
    nodextr.resize(nsample);
    nodexvr.resize(nsample);
    ndbegin.resize(nsample);
    jvr.resize(nsample);
    jtr.resize(nsample);
    jest.resize(nsample);
    isort.resize(nsample);
    itemp.resize(nsample);
    jinbag.resize(nsample);
    joobag.resize(nsample);
    idmove.resize(nsample);
    bestvar.resize(maxnode);
    nod.resize(maxnode);
    nodestatus.resize(maxnode);
    nodestart.resize(maxnode);
    nodestop.resize(maxnode);
    nodeclass.resize(maxnode);
    treemap.resize(2 * maxnode);        // (2, maxnode)
    catgoleft.resize(maxcat * maxnode); // (maxcat, maxnode)

    nc.resize(nclass);
    ncn.resize(nsample);
    igoleft.resize(nsample);
    igl.resize(nsample);
    itmp.resize(nsample);
    nodexb.resize(nsample);
    npcase.resize(nsample);
    ncount.resize(nsample);
    tcat.resize(nsample);
    icat.resize(nsample);

    // Proximity arrays (double precision)
    prox.resize(nsample * nsample);     // (nsample, nsample)
    proxsym.resize(nsample * nsample);  // (nsample, nsample)
    dwork.resize(nsample);
}

} // namespace rf
