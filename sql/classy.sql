create table {project}_variability (
       objectid text not null,
       datarelease integer not null,
       lcversion integer not null,
       firstdate double precision,
       lastdate double precision,
       obslength double precision,
       -- GLSP for each aperture
       glsp_sigclip real,
       glsp_timebinsec real,
       glsp_startp real,
       glsp_endp real,
       glsp_freqstep real,
       glsp_aep_bestperiod double precision[],
       glsp_aep_bestpeak real[],
       -- binned Dworetsky string length for each aperture
       dwsl_sigclip real,
       dwsl_timebinsec real,
       dwsl_aep_startp real,
       dwsl_aep_endp real,
       dwsl_aep_freqstep real,
       dwsl_aep_phasebinsize real,
       dwsl_aep_bestperiod real[],
       -- Stellingwerf phase dispersion minimization for each aperture
       spdm_sigclip real,
       spdm_timebinsec real,
       spdm_aep_startp real,
       spdm_aep_endp real,
       spdm_aep_freqstep real,
       spdm_aep_phasebinsize real,
       spdm_aep_bestperiod real[],
       -- Schwarzenberg-Cerny AoV for each aperture
       saov_sigclip real,
       saov_timebinsec real,
       saov_aep_startp real,
       saov_aep_endp real,
       saov_aep_freqstep real,
       saov_aep_phasebinsize real,
       saov_aep_bestperiod real[],
       -- LC features for each aperture using EPD mags
       lcfeature_sigclip real,
       lcfeature_timebinsec real,
       stetson_j_aep real[],
       stetson_k_aep real[],
       median_aep real[],
       wmean_aep real[],
       mad_aep real[],
       stdev_aep real[],
       amplitude_aep real[],
       skew_aep real[],
       kurtosis_aep real[],
       beyond1std_aep real[],
       linearfitslope_aep real[],
       p2pscatterovermad_aep real[],
       p2psqrdiffovervar_aep real[],
       magratio_aep real[],
       autocorr_aep_stdev real[],
       autocorr_aep_skew real[],
       autocorr_aep_kurtosis real[],
       autocorr_aep_stetsonk real[],
       -- LC features for each aperture using TFA mags
       stetson_j_atf real[],
       stetson_k_atf real[],
       median_atf real[],
       wmean_atf real[],
       mad_atf real[],
       stdev_atf real[],
       amplitude_atf real[],
       skew_atf real[],
       kurtosis_atf real[],
       beyond1std_atf real[],
       linearfitslope_atf real[],
       p2pscatterovermad_atf real[],
       p2psqrdiffovervar_atf real[],
       magratio_atf real[],
       autocorr_atf_stdev real[],
       autocorr_atf_skew real[],
       autocorr_atf_kurtosis real[],
       autocorr_atf_stetsonk real[],
       -- fourier fits for the LC using aep_000
       ffit_aep000_timebinsec real,
       ffit_aep000_sigclip real,
       ffit_aep000_order integer,
       ffit_aep000_period double precision,
       ffit_aep000_params double precision[],
       ffit_aep000_chisq real,
       ffit_aep000_redchisq real,
       ffit_aep000_fitepoch double precision,
       -- fourier fits for the LC using aep_002
       ffit_aep002_timebinsec real,
       ffit_aep002_sigclip real,
       ffit_aep002_order integer,
       ffit_aep002_period double precision,
       ffit_aep002_params double precision[],
       ffit_aep002_chisq real,
       ffit_aep002_redchisq real,
       ffit_aep002_fitepoch double precision,
       -- spline fits for LC using EPD mags in each aperture
       sfit_aep_timebinsec real,
       sfit_aep_sigclip real[],
       sfit_aep_period double precision[],
       sfit_aep_knots integer[],
       sfit_aep_chisq real[],
       sfit_aep_redchisq real[],
       sfit_aep_fitepoch double precision[],
       chisqratio_sfit_ffit real[],
       primary key (objectid, datarelease, lcversion)
);

create table {project}_lightcurves (

       -- primary database ID and keys
       objectid text not null,
       datarelease integer not null,

       -- object IDs
       hatid text,
       hatfield integer,
       hatfieldobjid integer,
       twomassid text,
       ucac4id text,

       -- object spatial information
       ra double precision not null,
       decl double precision not null,
       errbox box not null,
       pmra double precision,
       pmra_err double precision,
       pmdecl double precision,
       pmdecl_err double precision,

       -- object magnitudes and type
       jmag real,
       hmag real,
       kmag real,
       bmag real,
       vmag real,
       sdssg real,
       sdssr real,
       sdssi real,
       objecttags text,

       -- object detection information
       ndet integer not null default 0,
       stations text,
       network text,

       -- lightcurve information
       camfilters jsonb,
       lcapertures jsonb,
       bestaperture jsonb,
       lcfpath text not null default 'not collected',
       lastupdated timestamp with time zone not null default current_timestamp,
       lcversion integer not null default 0,
       lcserver text not null default 'ffffffff',

       -- variability information
       varobject boolean,
       vartags text,
       varperiodic boolean,
       varperiod double precision,
       varepochjd double precision,

       -- candidate information
       candidateid text,
       candidatetags text,
       candidateurl text,

       -- external information
       externalid text,
       externalurl text,
       externalarxiv text,
       externalads text,

       -- lightcurve aux info
       accessgroups text[] not null default '{"superuser", "researchgroup", "student"}',
       hatmosaicpath text,
       hatobservedfield text,
       primary key (objectid, datarelease, lcversion)
);
