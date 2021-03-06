package org.janelia.saalfeldlab.regionmerging;

import java.io.Serializable;
import java.lang.invoke.MethodHandles;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.function.BiConsumer;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.storage.StorageLevel;
import org.janelia.saalfeldlab.graph.UndirectedGraph;
import org.janelia.saalfeldlab.graph.edge.Edge;
import org.janelia.saalfeldlab.graph.edge.EdgeMerger;
import org.janelia.saalfeldlab.graph.edge.EdgeWeight;
import org.janelia.saalfeldlab.util.unionfind.HashMapStoreUnionFind;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import gnu.trove.list.array.TDoubleArrayList;
import gnu.trove.list.array.TLongArrayList;
import gnu.trove.map.hash.TIntObjectHashMap;
import gnu.trove.map.hash.TLongIntHashMap;
import gnu.trove.map.hash.TLongObjectHashMap;
import gnu.trove.set.hash.TIntHashSet;
import net.imglib2.util.Pair;
import scala.Tuple2;
import scala.Tuple3;

public class BlockedRegionMergingSpark
{

	public static Logger LOG = LoggerFactory.getLogger( MethodHandles.lookup().lookupClass() );

	public static class Data implements Serializable
	{

		private final TDoubleArrayList edges;

		private final HashMap< HashWrapper< long[] >, TIntHashSet > nonContractingEdges;

		public Data( final TDoubleArrayList edges, final HashMap< HashWrapper< long[] >, TIntHashSet > nonContractingEdges )
		{
			super();
			this.edges = edges;
			this.nonContractingEdges = nonContractingEdges;
		}

		public TDoubleArrayList edges()
		{
			return edges;
		}

		public HashMap< HashWrapper< long[] >, TIntHashSet > nonContractingEdges()
		{
			return nonContractingEdges;
		}

	}

	public static class Options
	{
		private double threshold;

		private StorageLevel persistenceLevel;

		private final int minimuMultiplicity = 0;

		public static Options options()
		{
			return new Options( 1.0, StorageLevel.MEMORY_ONLY() );
		}

		public Options( final double threshold, final StorageLevel persistenceLevel )
		{
			super();
			this.threshold = threshold;
			this.persistenceLevel = persistenceLevel;
		}

		public Options copy()
		{
			return new Options( threshold, persistenceLevel.clone() );
		}

		public Options threshold( final double threshold )
		{
			this.threshold = threshold;
			return this;
		}

		public Options persistenceLevel( final StorageLevel persistenceLevel )
		{
			this.persistenceLevel = persistenceLevel;
			return this;
		}

		public StorageLevel persistenceLevel()
		{
			return this.persistenceLevel.clone();
		}

		public double threshold()
		{
			return this.threshold;
		}
	}

	private final EdgeMerger merger;

	private final EdgeWeight edgeWeight;

	private final int factor;

	public BlockedRegionMergingSpark( final EdgeMerger merger, final EdgeWeight edgeWeight )
	{
		this( merger, edgeWeight, 2 );
	}

	public BlockedRegionMergingSpark( final EdgeMerger merger, final EdgeWeight edgeWeight, final int factor )
	{
		super();
		this.merger = merger;
		this.edgeWeight = edgeWeight;
		this.factor = factor;
	}

	public void agglomerate(
			final JavaSparkContext sc,
			final JavaPairRDD< HashWrapper< long[] >, Data > rdd,
			final BiConsumer< Integer, JavaPairRDD< HashWrapper< long[] >, Tuple2< TLongArrayList, HashMapStoreUnionFind > > > mergesLogger,
			final Options options )
	{

		final List< Object > unpersistList = new ArrayList<>();

		final Broadcast< Options > optionsBC = sc.broadcast( options );

		JavaPairRDD< HashWrapper< long[] >, Data > targetRdd = rdd;
		int iteration = 0;
		boolean hasReachedSingleBlock = false;
		final int dataSize = merger.dataSize();
		while ( !hasReachedSingleBlock )
		{
			hasReachedSingleBlock = targetRdd.count() == 1;
			LOG.info( targetRdd.count() + " blocks at iteration " + iteration );
			final int finalIteration = iteration;
			if ( LOG.isDebugEnabled() )
				targetRdd.map( t -> {
					LOG.debug( Arrays.toString( t._1().getData() ) + ": Starting agglomeration with " + new Edge( t._2().edges(), dataSize ).size() + " edges, and " + t._2().nonContractingEdges().values().stream().mapToInt( c -> c.size() ).sum() + " non-contracting edges." );
					return t;
				} ).count();
			final JavaPairRDD< HashWrapper< long[] >, Tuple3< Data, TLongArrayList, HashMapStoreUnionFind > > mergedEdgesWithMergeLog = createGraphAndContractMinimalEdges( targetRdd, optionsBC, merger, edgeWeight, finalIteration );
//			mergedEdgesWithMergeLog.map( t -> {
//				if ( t._2()._2().size() == 0 )
//					System.out.println( "ONLY ZERO MERGES WHY? " + Arrays.toString( t._1().getData() ) );
//				return true;
//			} ).count();
			mergedEdgesWithMergeLog.cache();
			final JavaPairRDD< HashWrapper< long[] >, Data > mergedEdges = mergedEdgesWithMergeLog.mapValues( p -> p._1() );
			mergesLogger.accept( iteration, mergedEdgesWithMergeLog.mapValues( p -> new Tuple2<>( p._2(), p._3() ) ) );

			final JavaPairRDD< HashWrapper< long[] >, Data > remapped = adjustPosition( mergedEdges, this.factor, edgeWeight.dataSize() );

			final JavaPairRDD< HashWrapper< long[] >, List< Data > > aggregated = aggregateAsList( remapped );

			final JavaPairRDD< HashWrapper< long[] >, Data > reduced = combineSubgraphs( aggregated, edgeWeight.dataSize() );

			final JavaPairRDD< HashWrapper< long[] >, Data > previous = targetRdd;
			targetRdd = reduced.persist( options.persistenceLevel );
			targetRdd.count();
			previous.unpersist();
			mergedEdgesWithMergeLog.unpersist();

			LOG.info( "Finished iteration " + iteration );
			++iteration;

		}

		if ( targetRdd != rdd )
			targetRdd.unpersist();

	}

	private static long[] adjustPosition( final long[] position, final long factor )
	{
		return Arrays.stream( position ).map( p -> p / factor ).toArray();
	}

	private static TLongObjectHashMap< TLongIntHashMap > createNodeEdgeMap( final Edge e )
	{
		final TLongObjectHashMap< TLongIntHashMap > nodeEdgeMap = new TLongObjectHashMap<>();
		for ( int edgeIndex = 0; edgeIndex < e.size(); ++edgeIndex )
		{
			e.setIndex( edgeIndex );
			if ( e.isValid() )
			{
				final long from = e.from();
				final long to = e.to();

				if ( !nodeEdgeMap.contains( from ) )
					nodeEdgeMap.put( from, new TLongIntHashMap() );
				if ( !nodeEdgeMap.contains( to ) )
					nodeEdgeMap.put( to, new TLongIntHashMap() );

				final TLongIntHashMap fromMap = nodeEdgeMap.get( from );
				final TLongIntHashMap toMap = nodeEdgeMap.get( to );

				if ( fromMap.contains( to ) )
				{
					assert toMap.contains( from ): "Map inconsistency!";
					e.setObsolete();
				}
				else
				{
					assert !toMap.contains( from ): "Map inconsistency!";
					fromMap.put( to, edgeIndex );
					toMap.put( from, edgeIndex );
				}
			}
		}
		return nodeEdgeMap;
	}

	private static JavaPairRDD< HashWrapper< long[] >, Data > adjustPosition( final JavaPairRDD< HashWrapper< long[] >, Data > input, final int factor, final int dataSize )
	{
		return input.mapToPair( t -> adjustPosition( t, factor, dataSize ) );
	}

	private static Tuple2< HashWrapper< long[] >, Data > adjustPosition( final Tuple2< HashWrapper< long[] >, Data > t, final int factor, final int dataSize )
	{
		final long[] key = adjustPosition( t._1().getData(), factor );
		final HashWrapper< long[] > hashedKey = new HashWrapper<>( key, t._1().getHash(), t._1().getEquals() );
		final Data data = t._2();
		final HashMap< HashWrapper< long[] >, TIntHashSet > nonContractingEdges = new HashMap<>();
		final TIntObjectHashMap< HashWrapper< long[] > > inverseNonContractingEdges = new TIntObjectHashMap<>();

		data.nonContractingEdges.forEach( ( k, v ) -> {
			final long[] adjustedKey = adjustPosition( k.getData(), factor );
			final HashWrapper< long[] > wrapped = new HashWrapper<>( adjustedKey, k.getHash(), k.getEquals() );
			if ( !Arrays.equals( adjustedKey, key ) ) {
				nonContractingEdges.put( wrapped, new TIntHashSet() );
//				nonContractingEdges.put( new HashWrapper<>( adjustedKey, k.getHash(), k.getEquals() ), v );
				v.forEach( entry -> {
					inverseNonContractingEdges.put( entry, wrapped );
					return true;
				} );
			}
		} );



		final TDoubleArrayList newEdgeStore = new TDoubleArrayList();
		final Edge newEdge = new Edge( newEdgeStore, dataSize );
		final Edge oldEdge = new Edge( data.edges, dataSize );

		for ( int i = 0; i < oldEdge.size(); ++i )
		{
			oldEdge.setIndex( i );
			if ( oldEdge.isValid() )
			{
				newEdge.setIndex( newEdge.size() - 1 );
				newEdge.add( oldEdge );
				final HashWrapper< long[] > wrapped = inverseNonContractingEdges.get( i );
				if ( wrapped != null )
					nonContractingEdges.get( wrapped ).add( newEdge.size() - 1 );
			}
		}


		// keep non-contracting edges when not in the same block (after
		// adjusting coordinates)

		return new Tuple2<>( hashedKey, new Data( newEdgeStore, nonContractingEdges ) );
	}

	private static final JavaPairRDD< HashWrapper< long[] >, Tuple3< Data, TLongArrayList, HashMapStoreUnionFind > > createGraphAndContractMinimalEdges(
			final JavaPairRDD< HashWrapper< long[] >, Data > input,
			final Broadcast< Options > optionsBC,
			final EdgeMerger merger,
			final EdgeWeight edgeWeight,
			final int iteration )
	{
		return input.mapValues( new Contract( optionsBC, merger, edgeWeight ) );
	}

	public static class Contract implements Function< Data, Tuple3< Data, TLongArrayList, HashMapStoreUnionFind > >
	{

		private final Broadcast< Options > optionsBC;

		private final EdgeMerger merger;

		private final EdgeWeight weight;

		public Contract( final Broadcast< Options > optionsBC, final EdgeMerger merger, final EdgeWeight weight )
		{
			super();
			this.optionsBC = optionsBC;
			this.merger = merger;
			this.weight = weight;
		}

		@Override
		public Tuple3< Data, TLongArrayList, HashMapStoreUnionFind > call( final Data data ) throws Exception
		{
			return createGraphAndContractMinimalEdges( data, optionsBC, merger, weight );
		}

	}

	private static Tuple3< Data, TLongArrayList, HashMapStoreUnionFind > createGraphAndContractMinimalEdges( final Data data, final Broadcast< Options > optionsBC, final EdgeMerger merger, final EdgeWeight edgeWeight )
	{
		final TIntHashSet nonContractingEdges = new TIntHashSet();
		final Options opt = optionsBC.getValue();
		data.nonContractingEdges.values().forEach( nonContractingEdges::addAll );
		final UndirectedGraph g = new UndirectedGraph( data.edges, createNodeEdgeMap( new Edge( data.edges, edgeWeight.dataSize() ) ), edgeWeight.dataSize() );
		final Pair< TLongArrayList, HashMapStoreUnionFind > mergesAndMapping = RegionMerging.mergeLocallyMinimalEdges( g, merger, edgeWeight, opt.threshold, opt.minimuMultiplicity, nonContractingEdges );
		final TLongArrayList merges = mergesAndMapping.getA();
		final HashMapStoreUnionFind mapping = mergesAndMapping.getB();

		final Edge e = new Edge( data.edges, edgeWeight.dataSize() );

		for ( int i = 0; i < e.size(); ++i )
		{
			e.setIndex( i );
			if ( e.isValid() )
			{
				e.from( mapping.findRoot( e.from() ) );
				e.to( mapping.findRoot( e.to() ) );
			}
		}

		return new Tuple3<>( data, merges, mapping );
	}

	private static JavaPairRDD< HashWrapper< long[] >, Data > combineSubgraphs( final JavaPairRDD< HashWrapper< long[] >, List< Data > > input, final int dataSize )
	{
		return input.mapValues( dataList -> combineSubgraphs( dataList, dataSize ) );
	}

	private static Data combineSubgraphs( final List< Data > dataList, final int dataSize )
	{

		final TDoubleArrayList edgeStore = new TDoubleArrayList();
		final Edge e = new Edge( edgeStore, dataSize );
		final HashMap< HashWrapper< long[] >, TIntHashSet > nonContractingEdges = new HashMap<>();

		for ( final Data d : dataList )
		{

			// invert mapping such that we can check if edge is non-contracting
			// more easily
			final TIntObjectHashMap< HashWrapper< long[] > > inverseNonContractingEdges = new TIntObjectHashMap<>();
			d.nonContractingEdges.forEach( ( k, v ) -> v.forEach( edgeIndex -> {
				inverseNonContractingEdges.put( edgeIndex, k );
				return true;
			} ) );

			// copy old edges into new edge list, and add to non-contracting
			// edges if applicable
			final Edge oldEdge = new Edge( d.edges, dataSize );
			for ( int i = 0; i < oldEdge.size(); ++i )
			{
				oldEdge.setIndex( i );
				if ( oldEdge.isValid() )
				{
					final int newEdgeIndex = e.size();
					e.add( oldEdge );
					if ( inverseNonContractingEdges.contains( i ) )
					{
						final HashWrapper< long[] > key = inverseNonContractingEdges.get( i );
						if ( !nonContractingEdges.containsKey( key ) )
							nonContractingEdges.put( key, new TIntHashSet() );
						nonContractingEdges.get( key ).add( newEdgeIndex );
					}
				}
			}

		}

		return new Data( edgeStore, nonContractingEdges );
	}

	private static JavaPairRDD< HashWrapper< long[] >, List< Data > > aggregateAsList( final JavaPairRDD< HashWrapper< long[] >, Data > remapped )
	{
		final JavaPairRDD< HashWrapper< long[] >, List< Data > > aggregated = remapped.aggregateByKey( new ArrayList<>(),
				( list, data ) -> {
					list.add( data );
					return list;
				},
				( list1, list2 ) -> {
					final ArrayList< Data > list = new ArrayList<>();
					list.addAll( list1 );
					list.addAll( list2 );
					return list;
				} );
		return aggregated;
	}

}
